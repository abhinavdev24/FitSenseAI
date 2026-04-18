import Link from 'next/link'

export default function MeetTeam() {
  return (
    <section className="py-16 px-6 bg-blue-50 text-center">
      <h2 className="text-2xl sm:text-3xl font-bold text-blue-900 mb-3" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
        Built by a team of graduate students
      </h2>
      <p className="text-slate-600 mb-8 max-w-xl mx-auto">
        Six engineers and researchers who turned a machine learning project into a real, working AI fitness coach.
      </p>
      <Link
        href="/about"
        className="inline-block px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-full text-lg transition-all shadow-lg shadow-blue-200"
      >
        Meet the Team →
      </Link>
    </section>
  )
}
