import { create } from 'zustand'

interface OpenAiTokenState {
  openAIKey: string;
  selectedModel: string;
  setOpenAIKey: (token: string) => void;
  setSelectedModel: (model: string) => void;
}

export const useOpenAiTokenStore = create<OpenAiTokenState>((set) => ({
  openAIKey: "",
  selectedModel: "gpt-5.1",
  setOpenAIKey: (token: string) => set({ openAIKey: token }),
  setSelectedModel: (model: string) => set({ selectedModel: model }),
}))
