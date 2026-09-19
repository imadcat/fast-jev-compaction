/**
 * The omp-side adapter for fast-jev-compaction.
 *
 * omp's `session_before_compact` hook can replace the built-in summarizer by
 * returning `{ compaction }`. fast-jev-compaction does not write a summary: it
 * asks Jev which tool calls and results are still needed, drops the rest, and
 * keeps every surviving message verbatim.
 *
 * omp commits a compaction as a `summary` plus a `firstKeptEntryId` boundary,
 * so the verbatim result has to be expressed in those terms. This adapter
 * therefore embeds the surviving transcript — rebuilt as text, in order — as
 * the compaction `summary`, and sets `firstKeptEntryId` to the boundary omp
 * chose. The dropped calls and results are gone; everything kept is byte-exact.
 *
 * See `README.md` for the design note and the trade-off this implies.
 */

import { compact, reductionRatio } from "../src/compact.js";
import type { CompactOptions, CompactResult, Message } from "../src/types.js";

/** fje-omp: default TypeSafe System One endpoint. */
const DEFAULT_MODEL = "jev-latest";

interface ContentBlock {
	type?: string;
	text?: string;
	thinking?: string;
	id?: string;
	name?: string;
	arguments?: Record<string, unknown>;
}

interface EntryMessage {
	role?: string;
	content?: unknown;
	toolCallId?: string;
	toolName?: string;
	isError?: boolean;
}

interface BranchEntry {
	type?: string;
	id?: string;
	message?: EntryMessage;
	summary?: string;
	customType?: string;
	content?: unknown;
}

export interface PluginOptions {
	/** Which endpoint answers the Jev questions. Default `typesafe`. */
	provider?: "typesafe" | "openrouter";
	apiKey?: string;
	model?: string;
	baseUrl?: string;
	keepThreshold?: number;
	preserveRecentMessages?: number;
	maxStateTokens?: number;
	maxRequestTokens?: number;
	truncateHeadChars?: number;
	goal?: string;
	/** Below this character-reduction ratio the built-in summary runs instead. */
	minReductionRatio?: number;
}

const HOOK_DEFAULTS = {
	model: DEFAULT_MODEL,
	minReductionRatio: 0.25,
};

/** `ctx` and `pi` as much of omp's extension surface as this adapter touches. */
export interface ExtensionSurface {
	on(
		event: "session_before_compact",
		handler: (
			event: {
				preparation: {
					firstKeptEntryId?: string;
					tokensBefore?: number;
					settings?: Record<string, unknown>;
				};
				branchEntries: readonly BranchEntry[];
				signal?: AbortSignal;
			},
			ctx: { env?: { get(name: string): Promise<string | undefined> }; ui?: UI },
		) => Promise<unknown>,
	): void;
	logger?: { warn(message: string): void; info?(message: string): void };
}

export interface UI {
	notify?(message: string, level?: string): void;
	setStatus?(key: string, text: string): void;
	log?(message: string): void;
}

/** Plain text of a message payload; images are noted, thinking is preserved. */
function textFromContent(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	const parts: string[] = [];
	for (const block of content as ContentBlock[]) {
		if (block?.type === "text" && typeof block.text === "string") parts.push(block.text);
		else if (block?.type === "thinking" && typeof block.thinking === "string") {
			parts.push(block.thinking);
		} else if (block?.type === "image") parts.push("[image]");
	}
	return parts.join("\n");
}

interface MessageOrigin {
	/** Index into the omp entry list this message was built from. */
	entryIndex: number;
	entryId: string | undefined;
	/** Every `tool_use_id` assigned to this message's tool uses, in order. */
	useIds: string[];
}

interface Mapped {
	messages: Message[];
	origins: MessageOrigin[];
}

/**
 * Maps omp's branch entries 1:1 onto the library's transcript.
 *
 * One omp entry becomes one `Message`, preserving the entry granularity the
 * library's pinning relies on: it pins the newest `preserveRecentMessages`
 * messages by index and the first message always. Folding results into their
 * assistant message (the obvious-looking optimisation) collapses a transcript
 * to a handful of messages and pins all of it, so nothing is ever a candidate.
 *
 * The library's own `applyDecisions` expects this split shape: it handles
 * `toolUses` and `toolResults` per message and pairs them by `tool_use_id`
 * across messages, which is also how it reads Claude Code transcripts.
 *
 *   assistant entry   -> Message{ role: "assistant", text, toolUses: [...] }
 *   toolResult entry  -> Message{ role: "user", toolResults: [{ tool_use_id, text, isError }] }
 */
export function mapBranchEntries(entries: readonly BranchEntry[]): Mapped {
	const messages: Message[] = [];
	const origins: MessageOrigin[] = [];

	for (let entryIndex = 0; entryIndex < entries.length; entryIndex++) {
		const entry = entries[entryIndex]!;
		if (entry.type !== "message") continue;
		const message = entry.message;
		const role = message?.role;
		if (!role) continue;

		if (role === "toolResult") {
			// omp names the field `toolCallId`; the library pairs on `tool_use_id`.
			messages.push({
				role: "user",
				text: "",
				toolUses: [],
				toolResults: [
					{
						tool_use_id: typeof message?.toolCallId === "string" ? message.toolCallId : "",
						text: textFromContent(message?.content),
						isError: message?.isError === true,
					},
				],
			});
			origins.push({ entryIndex, entryId: entry.id, useIds: [] });
			continue;
		}

		const blocks = Array.isArray(message?.content) ? (message.content as ContentBlock[]) : [];
		const toolUses: Message["toolUses"] = [];
		const useIds: string[] = [];
		for (const block of blocks) {
			if (block?.type !== "toolCall") continue;
			const id =
				typeof block.id === "string" && block.id
					? block.id
					: `call_${entry.id ?? entryIndex}_${useIds.length}`;
			useIds.push(id);
			toolUses.push({
				tool_use_id: id,
				tool: typeof block.name === "string" ? block.name : "tool",
				input: block.arguments ?? {},
			});
		}
		messages.push({
			role: role === "assistant" ? "assistant" : "user",
			text: textFromContent(message?.content),
			toolUses,
		});
		origins.push({ entryIndex, entryId: entry.id, useIds });
	}

	return { messages, origins };
}

/**
 * Renders the post-compaction transcript as text. Called and surviving results
 * are printed in order; the library already removed everything Jev let go, so
 * what remains is verbatim.
 */
export function renderVerbatimTranscript(
	messages: readonly Message[],
	truncationNotice: string,
): string {
	const blocks: string[] = [];
	for (const message of messages) {
		if (message.text.trim().length > 0) {
			blocks.push(`### ${message.role}\n${message.text}`);
		}
		for (const use of message.toolUses) {
			blocks.push(`#### tool call: ${use.tool}\n\`\`\`json\n${safeJson(use.input)}\n\`\`\``);
		}
		for (const result of message.toolResults ?? []) {
			blocks.push(`#### tool result${result.isError ? " (error)" : ""}\n${result.text}`);
		}
	}
	if (blocks.length === 0) return truncationNotice;
	return blocks.join("\n\n");
}

function safeJson(value: unknown): string {
	try {
		return JSON.stringify(value, null, 2);
	} catch {
		return "[unserializable]";
	}
}

export function resolveOptions(options: PluginOptions): CompactOptions {
	const resolved: CompactOptions = {};
	if (typeof options.goal === "string" && options.goal) resolved.goal = options.goal;
	for (const key of [
		"keepThreshold",
		"preserveRecentMessages",
		"maxStateTokens",
		"maxRequestTokens",
		"truncateHeadChars",
	] as const) {
		const value = options[key];
		if (typeof value === "number" && Number.isFinite(value)) resolved[key] = value;
	}
	return resolved;
}

export function summaryLine(result: CompactResult): string {
	const { stats } = result;
	const parts = [
		stats.kept > 0 ? `${stats.kept} kept` : "",
		stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : "",
		stats.callsDropped > 0 ? `${stats.callsDropped} calls dropped` : "",
		stats.pinned > 0 ? `${stats.pinned} pinned` : "",
	].filter(Boolean);
	return `${Math.round(reductionRatio(result) * 100)}% reduction; ${parts.join(", ") || "no tool calls"}; state ~${stats.stateTokens} tokens (${stats.stateStage}) in ${stats.requests} request(s)`;
}

/**
 * Both endpoints speak the same protocol — `{ model, state, questions }` with
 * `noul`/`choice`/`score` questions — so the provider choice is only a base
 * URL, a model name, and which env var holds the key.
 *
 * OpenRouter is the useful one when a key already exists: the Jev models there
 * are `decisions` models, which the chat/completions wire rejects, but which
 * this endpoint (and therefore this plugin) speaks natively.
 */
export const PROVIDERS = {
	typesafe: {
		label: "TypeSafe System One",
		baseUrl: "https://api.typesafe.ai/v1/systemone",
		model: "jev-latest",
		keyVar: "TYPESAFE_API_KEY",
	},
	openrouter: {
		label: "OpenRouter decisions",
		baseUrl: "https://openrouter.ai/api/alpha/decisions",
		model: "typesafe/jev-1.13",
		keyVar: "OPENROUTER_API_KEY",
	},
} as const;

export type ProviderName = keyof typeof PROVIDERS;

export function resolveConfig(options: PluginOptions) {
	const provider: ProviderName = options.provider === "openrouter" ? "openrouter" : "typesafe";
	const spec = PROVIDERS[provider];
	return {
		provider,
		providerLabel: spec.label,
		keyVar: spec.keyVar,
		model: options.model ?? spec.model,
		baseUrl: options.baseUrl ?? spec.baseUrl,
		minReductionRatio:
			typeof options.minReductionRatio === "number" && Number.isFinite(options.minReductionRatio)
				? options.minReductionRatio
				: HOOK_DEFAULTS.minReductionRatio,
	};
}

/**
 * Resolves the provider's key from the plugin option, then the process
 * environment, then omp's own env lookup. `keyVar` comes from the resolved
 * provider, so switching providers switches the variable that is read.
 */
export async function resolveApiKey(
	options: PluginOptions,
	ctx: { env?: { get(name: string): Promise<string | undefined> } },
	keyVar: string = PROVIDERS.typesafe.keyVar,
): Promise<string | undefined> {
	if (typeof options.apiKey === "string" && options.apiKey.length > 0) return options.apiKey;
	const fromProcess = process.env[keyVar];
	if (fromProcess) return fromProcess;
	if (ctx.env) {
		const fromHost = await ctx.env.get(keyVar);
		if (fromHost) return fromHost;
	}
	return undefined;
}

export { compact, reductionRatio };
export type { CompactResult, Message };
