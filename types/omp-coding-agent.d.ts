/**
 * Minimal ambient declaration for the omp extension API surface this plugin
 * touches. The runtime injects the real module; this exists so the port can be
 * typechecked without a source checkout of omp.
 */

declare module "@oh-my-pi/pi-coding-agent" {
	export interface ExtensionUI {
		notify?(message: string, level?: "info" | "warning" | "error"): void;
		setStatus?(key: string, text: string): void;
		log?(message: string): void;
	}

	export interface ExtensionContext {
		ui?: ExtensionUI;
		cwd?: string;
		env?: { get(name: string): Promise<string | undefined> };
		sessionManager?: {
			getSessionFile?(): string | undefined;
			getSessionId?(): string;
			getBranch?(): readonly unknown[];
		};
	}

	export interface SessionBeforeCompactPreparation {
		firstKeptEntryId?: string;
		tokensBefore?: number;
		settings?: Record<string, unknown>;
	}

	export interface SessionBeforeCompactEvent {
		type: "session_before_compact";
		preparation: SessionBeforeCompactPreparation;
		branchEntries: readonly unknown[];
		customInstructions?: string;
		signal?: AbortSignal;
	}

	export interface CompactionResultPayload {
		summary: string;
		shortSummary?: string;
		firstKeptEntryId?: string;
		tokensBefore: number;
		details?: unknown;
	}

	export interface ExtensionAPI {
		on(
			event: "session_before_compact",
			handler: (
				event: SessionBeforeCompactEvent,
				ctx: ExtensionContext,
			) => Promise<{ compaction?: CompactionResultPayload; cancel?: boolean } | undefined | void>,
		): void;
		on(event: string, handler: (event: never, ctx: ExtensionContext) => unknown): void;
		setLabel?(label: string): void;
		logger?: { warn(message: string): void; info?(message: string): void };
	}

	export type ExtensionFactory = (pi: ExtensionAPI, options?: unknown) => void | Promise<void>;
}
