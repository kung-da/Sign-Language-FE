import { Footer } from "./components/layout/Footer";
import { Navbar } from "./components/layout/Navbar";
import { AboutSection } from "./components/sections/AboutSection";
import { ArchitectureSection } from "./components/sections/ArchitectureSection";
import { DemoSection } from "./components/sections/DemoSection";
import { FeaturesSection } from "./components/sections/FeaturesSection";
import { HeroSection } from "./components/sections/HeroSection";
import { MetricsSection } from "./components/sections/MetricsSection";
import { PipelineSection } from "./components/sections/PipelineSection";

export default function App() {
  return (
    <div className="min-h-screen overflow-hidden">
      <Navbar />
      <main>
        <HeroSection />
        <DemoSection />
        <PipelineSection />
        <ArchitectureSection />
        <MetricsSection />
        <FeaturesSection />
        <AboutSection />
      </main>
      <Footer />
    </div>
  );
}
