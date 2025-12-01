"use client";
import { createContext, useContext, useState, ReactNode } from "react";

interface Settings {
  temperature: number;
  maxTokens: number;
}

interface SettingsContextType {
  settings: Settings;
  updateTemperature: (temperature: number) => void;
  updateMaxTokens: (maxTokens: number) => void;
}

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>({
    temperature: 0.7, // default value
    maxTokens: 100, // default value
  });

  const updateTemperature = (temperature: number) => {
    setSettings((prev) => ({ ...prev, temperature }));
  };

  const updateMaxTokens = (maxTokens: number) => {
    setSettings((prev) => ({ ...prev, maxTokens }));
  };

  return (
    <SettingsContext.Provider value={{ settings, updateTemperature, updateMaxTokens }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings() {
  const context = useContext(SettingsContext);
  if (!context) {
    throw new Error("useSettings must be used within SettingsProvider");
  }
  return context;
}
