# Local PDF TTS development

Use this guidance for work on `local-tts`.

1. Preserve the Apple-Silicon-first, local-only scope. Prefer MLX for Kokoro;
   add alternate inference runtimes only behind the backend interface.
2. Keep the CLI page-oriented and resume-safe. A chunk is durable only after
   its PCM bytes and JSONL checkpoint entry have both been written.
3. Make throughput measurements comparable: record generated audio duration,
   wall time, real-time factor, profile, backend, chunk size, system peak RSS,
   and MLX allocator telemetry when applicable.
4. Avoid unbounded queues. For MLX, apply `mlx.set_memory_limit` before model
   load and describe it honestly as an MLX allocation guideline, not a macOS
   process-wide cgroup.
5. Do not "clean" away technical content by default. Expose cleanup decisions
   through options and test the text transformations independently from TTS.
6. Use `pypdf`'s layout-preserving extraction beside regular text extraction for
   technical-block detection. Require a code/schema signal or a multi-row
   table run; never classify dotted contents leaders as a table. Carry embedded
   PDF metadata into the job manifest and supported audio tags.
7. Verify new backends with a short real PDF synthesis and `ffprobe`, plus unit
   tests that do not require downloading model weights. For layout changes,
   also inspect `local-tts book.pdf --metadata` and at least one real code or
   table page.
8. For optional LLM summaries, use a local MLX model, treat selected pages as
   primary evidence, cap all chapter context, and persist provenance and cost
   metrics. Unload the LLM before Kokoro synthesis and test planning without
   downloading model weights.
