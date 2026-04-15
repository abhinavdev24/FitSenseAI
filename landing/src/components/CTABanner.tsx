export default function CTABanner() {
  return (
    <section className="py-24 px-6 bg-blue-900">
      <div className="max-w-2xl mx-auto text-center">
        <div className="text-5xl mb-6">✦</div>
        <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
          Ready to train smarter?
        </h2>
        <p className="text-blue-200 text-lg mb-10">
          Get your personalized AI plan in minutes. Just install the APK, create an account, and answer a few questions — your first plan generates immediately.
        </p>
        <a
          href="/fitsense.apk"
          download
          className="inline-flex items-center gap-3 px-10 py-4 bg-white text-blue-900 hover:bg-blue-50 font-bold rounded-full text-lg transition-all shadow-xl"
        >
          <span className="text-xl">⬇</span> Download FitSense AI APK
        </a>
        <p className="mt-5 text-blue-300 text-sm">Android only · ~50 MB · Free</p>
      </div>
    </section>
  )
}
