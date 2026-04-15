import Image from 'next/image'

export default function Footer() {
  return (
    <footer className="py-8 px-6 bg-white border-t border-slate-200">
      <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Image src="/logo.svg" alt="FitSense AI logo" width={28} height={28} />
          <span className="text-slate-600 text-sm font-medium">FitSense AI</span>
        </div>
        <p className="text-slate-400 text-sm">© 2026 FitSense AI · Built with AI</p>
      </div>
    </footer>
  )
}
