import Image from 'next/image'
import Link from 'next/link'

export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 bg-white border-b border-slate-200 shadow-sm">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-3">
          <Image src="/logo.svg" alt="FitSense AI logo" width={36} height={36} />
          <span className="text-xl font-bold text-blue-900" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            FitSense AI
          </span>
        </Link>
        <div className="flex items-center gap-6">
          <Link href="/about" className="text-sm font-semibold text-slate-600 hover:text-blue-600 transition-colors hidden sm:block">
            About Us
          </Link>
          <a
            href="/fitsense.apk"
            download
            className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold rounded-full transition-all shadow-md shadow-blue-200 flex items-center gap-1.5"
          >
            <span>⬇</span> Download APK
          </a>
        </div>
      </div>
    </nav>
  )
}
