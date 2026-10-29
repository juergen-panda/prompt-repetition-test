# Prompt Repetition Test

Small CLI to test the effect of prompt repetition with non-reasoning models on structured output.

## How to Use

1. Run a server with a language model of your choice. For example, start a [llama.cpp](https://github.com/ggml-org/llama.cpp) server and call it via the OpenAI-compatible API:
```shell
./llama-server -hf unsloth/Qwen3.5-0.8B-GGUF:Q2_K_XL
```

2. Check out the code and run the CLI for evaluating the effect:
```shell
cargo run --release
```

Note: you need [Rust](https://rust-lang.org/) to compile the code.

Use log levels for further information. For example, `RUST_LOG=info cargo run --release` or `RUST_LOG=debug cargo run` outputs further details including LLM prompts and responses.
