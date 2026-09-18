/**
 * fast-jev-compaction for omp.
 *
 * Replaces omp's built-in compaction summary with a Jev decision pass:
 * every tool call and result is scored in one fast request, stale ones are
 * dropped or truncated, and everything kept stays verbatim.
 *
 * Registered as an omp extension through `package.json#omp.extensions`. The
 * `session_before_compact` hook returns `{ compaction }`, which omp commits in
 * place of its own summarizer (`fromExtension: true` on the compaction entry).
 *
 * Configuration, in precedence order:
 *   1. plugin options (see `.omp-plugin/plugin.json` and `omp plugin config`)
 *   2. `~/.omp/agent/config.yml` under `fastJevCompaction`
 *   3. environment: `FAST_JEV_PROVIDER`, `FAST_JEV_MODEL`, `FAST_JEV_BASE_URL`
 *
 * Provider selection lives in `./adapter.ts` (`PROVIDERS`): TypeSafe's System
 * One, or OpenRouter's decisions endpoint. Each reads its own key variable
 * (`TYPESAFE_API_KEY` / `OPENROUTER_API_KEY`), so a session started with
 * `OPENROUTER_API_KEY=... omp` needs no file to hold the secret.
 */

import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import {
	compact,
	mapBranchEntries,
	renderVerbatimTranscript,
	resolveApiKey,
	resolveConfig,
	resolveOptions,
	summaryLine,
	type PluginOptions,
} from "./adapter.js";
import type { CompactResult, JevAsker, Message } from "../src/types.js";
import { buildJevRequest, parseJevResponse } from "../src/request.js";

interface BranchEntryLike {
	type?: string;
	id?: string;
	message?: { role?: string; content?: unknown; toolCallId?: string; isError?: boolean };
	summary?: string;
}

const TRUNCATION_NOTICE =
	"[fast-jev-compaction: history compacted by Jev; every surviving message is verbatim]";

/** Reads the `fastJevCompaction` block out of omp's config.yml without a YAML dep. */
function readConfigFile(): PluginOptions {
	const path = join(homedir(), ".omp", "agent", "config.yml");
	let text: string;
	try {
		text = readFileSync(path, "utf8");
	} catch {
		return {};
	}
	const lines = text.split("\n");
	const start = lines.findIndex((line) => line.trim().startsWith("fastJevCompaction:"));
	if (start < 0) return {};
	const options: Record<string, string | number> = {};
	for (let i = start + 1; i < lines.length; i++) {
		const line = lines[i]!;
		if (!/^\s+\S/.test(line)) break;
		const match = /^\s+([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$/.exec(line);
		if (!match) continue;
		const key = match[1]!;
		const value = match[2]!.replace(/^["']|["']$/g, "");
		const asNumber = Number(value);
		options[key] = Number.isFinite(asNumber) && value !== "" ? asNumber : value;
	}
	return options as PluginOptions;
}

const ENV_KEYS: Record<string, string> = {
	provider: "FAST_JEV_PROVIDER",
	model: "FAST_JEV_MODEL",
	baseUrl: "FAST_JEV_BASE_URL",
};

/**
 * An `OPENROUTER_API_KEY` in the environment selects the OpenRouter provider,
 * since that is the only reason to have one and it removes a config step.
 * An explicit `provider` option always wins.
 */
function providerFromEnvironment(merged: PluginOptions): PluginOptions {
	if (merged.provider) return merged;
	if (process.env.OPENROUTER_API_KEY && !process.env.TYPESAFE_API_KEY) {
		return { ...merged, provider: "openrouter" };
	}
	return merged;
}

function mergedOptions(pluginOptions: PluginOptions): PluginOptions {
	const file = readConfigFile();
	const merged: PluginOptions = { ...file, ...pluginOptions };
	for (const [field, env] of Object.entries(ENV_KEYS)) {
		const value = process.env[env];
		if (value && !merged[field as keyof PluginOptions]) {
			(merged as Record<string, unknown>)[field] = value;
		}
	}
	return providerFromEnvironment(merged);
}

export default function fastJevCompaction(pi: ExtensionAPI, rawOptions?: unknown): void {
	const pluginOptions = (rawOptions ?? {}) as PluginOptions;
	const options = mergedOptions(pluginOptions);
	const config = resolveConfig(options);
	const compactOptions = resolveOptions(options);

	pi.setLabel?.("fast-jev-compaction");

	let active = 0;

	pi.on("session_before_compact", async (event, ctx) => {
		active += 1;
		try {
			const apiKey = await resolveApiKey(options, ctx as never, config.keyVar);
			if (!apiKey) {
				ctx.ui?.notify?.(
					`fast-jev-compaction: ${config.keyVar} is not set (provider ${config.provider}: ${config.providerLabel})`,
					"warning",
				);
				return undefined;
			}

			const entries = (event.branchEntries ?? []) as readonly BranchEntryLike[];
			const { messages } = mapBranchEntries(entries);

			const fetchFn = globalThis.fetch;
			const asker: JevAsker = {
				async ask(state, questions) {
					const request = buildJevRequest(
						{ apiKey, model: config.model, baseUrl: config.baseUrl },
						state,
						questions,
					);
					const response = await fetchFn(request.url, {
						method: request.method,
						headers: request.headers,
						body: request.body,
					});
					return parseJevResponse(response.status, response.ok, await response.text());
				},
			};

			const result: CompactResult = await compact(messages, asker, compactOptions);
			const reduction = (result.stats.charsBefore - result.stats.charsAfter) / (result.stats.charsBefore || 1);
			if (reduction < config.minReductionRatio) {
				ctx.ui?.notify?.(
					`fast-jev-compaction: below ${Math.round(config.minReductionRatio * 100)}% minimum, using built-in summary (${summaryLine(result)})`,
					"info",
				);
				return undefined;
			}

			const transcript = renderVerbatimTranscript(result.messages, TRUNCATION_NOTICE);
			const preparation = event.preparation ?? {};
			ctx.ui?.notify?.(`fast-jev-compaction: ${summaryLine(result)}`, "info");

			return {
				compaction: {
					summary: transcript,
					shortSummary: `Jev compaction: ${summaryLine(result)}`,
					firstKeptEntryId: preparation.firstKeptEntryId,
					tokensBefore: preparation.tokensBefore ?? 0,
					details: {},
				},
			};
		} catch (error) {
			ctx.ui?.notify?.(
				`fast-jev-compaction: falling back to built-in summary (${error instanceof Error ? error.message : String(error)})`,
				"warning",
			);
			return undefined;
		} finally {
			active -= 1;
			void active;
		}
	});
}
