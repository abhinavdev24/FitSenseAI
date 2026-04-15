const features = [
  {
    icon: '🎯',
    title: 'Personalized Plans',
    desc: 'Tell us your goal, available equipment, and any constraints. FitSense AI builds a structured weekly program tailored to you — not a generic template.',
  },
  {
    icon: '🔄',
    title: 'AI Adaptation',
    desc: 'Your plan evolves with you. Each week the AI reviews your logged performance and adjusts volume, intensity, and exercises accordingly.',
  },
  {
    icon: '📋',
    title: 'Workout Logging',
    desc: 'Log every session — sets, reps, weight, and RIR (reps in reserve). A clean interface that stays out of your way while you train.',
  },
  {
    icon: '📊',
    title: 'Daily Check-ins',
    desc: 'Track sleep, calorie intake, and body weight. These signals feed the adaptation engine so your plan reflects real recovery, not just planned load.',
  },
  {
    icon: '🛡️',
    title: 'Safety-Aware Coaching',
    desc: 'Injury flags and medical constraints are respected at every step. The AI uses conservative language and escalates when pain or risk is mentioned.',
  },
  {
    icon: '💬',
    title: 'AI Coach Chat',
    desc: 'Ask anything about your training — technique, scheduling, substitutions. The AI stays scoped to fitness coaching and gives actionable answers.',
  },
]

export default function Features() {
  return (
    <section className="py-24 px-6 bg-white">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-blue-900 mb-4" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            Everything you need to train smarter
          </h2>
          <p className="text-slate-600 text-lg max-w-xl mx-auto">
            One app that plans, tracks, and adapts — no spreadsheets, no guesswork.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((f) => (
            <div key={f.title} className="bg-white border border-slate-200 rounded-2xl p-7 shadow-sm hover:shadow-md hover:-translate-y-1 transition-all">
              <div className="text-4xl mb-4">{f.icon}</div>
              <h3 className="text-lg font-semibold text-slate-900 mb-2" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>{f.title}</h3>
              <p className="text-slate-600 text-sm leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
