// This file was generated by [ts-rs](https://github.com/Aleph-Alpha/ts-rs). Do not edit this file manually.
import type { DummyProvider } from "./DummyProvider";
import type { OpenAIProvider } from "./OpenAIProvider";

export type EmbeddingProviderConfig =
  | { OpenAI: OpenAIProvider }
  | { Dummy: DummyProvider };
