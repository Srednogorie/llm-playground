import { MyAssistant } from "@/components/MyAssistant";
import { Navigation } from "@/components/Navigation";

export default function Home() {
  return (
    <main className="h-dvh">
      <div className="flex w-full flex-grow flex-col items-center justify-center">
        <Navigation />
      </div>
      <MyAssistant />
    </main>
  );
}
