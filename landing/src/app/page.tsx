import Navbar from '@/components/Navbar'
import Hero from '@/components/Hero'
import AppScreenshots from '@/components/AppScreenshots'
import VideoSection from '@/components/VideoSection'
import Features from '@/components/Features'
import About from '@/components/About'
import Personas from '@/components/Personas'
import MeetTeam from '@/components/MeetTeam'
import CTABanner from '@/components/CTABanner'
import Footer from '@/components/Footer'

export default function Home() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <AppScreenshots />
        <VideoSection />
        <Features />
        <About />
        <Personas />
        <MeetTeam />
        <CTABanner />
      </main>
      <Footer />
    </>
  )
}
