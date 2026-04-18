export default function Hero() {
  return (
    <section className="bg-gradient-to-br from-slate-50 to-blue-50 py-28 px-6">
      <div className="max-w-3xl mx-auto text-center">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-blue-100 text-blue-700 rounded-full text-sm font-medium mb-8">
          <span>✦</span><span>AI-Powered Fitness Coaching</span><span>✦</span>
        </div>
        <h1
          className="text-5xl sm:text-6xl font-bold leading-tight mb-6 bg-gradient-to-r from-blue-900 via-blue-600 to-blue-400 bg-clip-text text-transparent"
          style={{ fontFamily: "'Space Grotesk', sans-serif" }}
        >
          Your AI Fitness Coach
        </h1>
        <p className="text-lg sm:text-xl text-slate-600 leading-relaxed mb-10 max-w-2xl mx-auto">
          Personalized workout plans that adapt week-over-week — built on a fine-tuned AI model trained on real coaching expertise.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <a
            href="/fitsense.apk"
            download
            className="px-10 py-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-full text-lg transition-all shadow-lg shadow-blue-200 hover:shadow-blue-300 hover:-translate-y-0.5 flex items-center gap-2 justify-center"
          >
            <span>⬇</span> Download FitSense AI
          </a>
          <a
            href="#about"
            className="px-8 py-4 bg-white hover:bg-slate-50 text-blue-900 font-semibold rounded-full text-lg border border-slate-200 transition-colors"
          >
            Learn More
          </a>
        </div>
        <p className="mt-5 text-sm text-slate-400">Android APK · Free · No account required to install</p>
      </div>
    </section>
  )
}
