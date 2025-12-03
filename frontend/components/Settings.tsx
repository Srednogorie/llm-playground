"use client";
import { useSettings } from "@/lib/SettingsContext";

export function Settings() {
  const { settings, updateModel, updateTemperature, updateMaxTokens } = useSettings();

  return (
    <div className="mb-4 ml-4 rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex items-center justify-between mb-6">
        <label htmlFor="model-select" className="text-sm font-medium text-gray-700">
          Model
        </label>
        <select
          id="model-select"
          value={settings.model}
          onChange={(e) => updateModel(e.target.value)}
          className="rounded border border-gray-300 bg-white px-3 py-1 text-sm text-gray-700 shadow-sm hover:border-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          <option value="gpt-4.1-nano">GPT 4.1 Nano</option>
          <option value="claude-3-haiku-20240307">Haiku 3</option>
        </select>
      </div>
      <div className="flex items-center justify-between mb-6">
        <label htmlFor="temperature" className="text-sm font-medium text-gray-700">
          Temperature
        </label>
        <div className="flex">
          <input
            id="temperature"
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={settings.temperature}
            onChange={(e) => updateTemperature(parseFloat(e.target.value))}
            className="w-32"
          />
          <span className="w-8 text-sm text-gray-600">{settings.temperature.toFixed(1)}</span>
        </div>
      </div>
      <div className="flex items-center justify-between">
        <label htmlFor="max-tokens" className="text-sm font-medium text-gray-700">
          Max Tokens
        </label>
        <div className="flex">
          <input
            id="max-tokens"
            type="range"
            min="10"
            max="2000"
            step="1"
            value={settings.maxTokens}
            onChange={(e) => updateMaxTokens(parseInt(e.target.value))}
            className="w-32"
          />
          <span className="w-8 text-sm text-gray-600">{settings.maxTokens.toFixed(1)}</span>
        </div>
      </div>
    </div>
  );
}
