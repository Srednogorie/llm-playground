import { MyAssistant } from "@/components/MyAssistant";
import { Navigation } from "@/components/Navigation";
import { Settings } from "@/components/Settings";
import { SettingsProvider } from "@/lib/SettingsContext";

export default function Home() {
  return (
    <SettingsProvider>
      <main className="h-dvh flex flex-col">
        <div className="flex w-full flex-col items-center justify-center m-6">
          <Navigation />
        </div>
        <div className="flex flex-row flex-1">
          <div className="flex-1">
            <Settings />
          </div>
          <div className="flex-2 flex flex-col justify-end overflow-hidden">
            <MyAssistant />
          </div>
        </div>
      </main>
    </SettingsProvider>
  );
}
