const personas = [
  {
    icon: '🌱',
    title: 'Beginner Lifter',
    subtitle: 'Just getting started',
    desc: 'Wants a simple, structured plan that builds confidence. FitSense AI explains the why behind each exercise and keeps progressions manageable.',
    tags: ['Clear structure', 'Guided progressions', 'No guesswork'],
  },
  {
    icon: '💪',
    title: 'Intermediate Lifter',
    subtitle: 'Chasing the next level',
    desc: 'Needs intelligent progression logic and tweaks based on actual performance — not the same plan week after week. Adaptation is the key feature.',
    tags: ['Smart progression', 'Performance-driven', 'Weekly adaptation'],
  },
  {
    icon: '🩺',
    title: 'Constraint-Heavy User',
    subtitle: 'Injuries, equipment limits, or medical conditions',
    desc: 'Has specific constraints that most apps ignore. FitSense AI respects injury flags, adjusts around equipment limitations, and uses conservative safety language.',
    tags: ['Injury-aware', 'Equipment flexibility', 'Safety-first'],
  },
]

export default function Personas() {
  return (
    <section className="py-24 px-6 bg-white">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-blue-900 mb-4" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            Who it&apos;s for
          </h2>
          <p className="text-slate-600 text-lg max-w-xl mx-auto">
            Whether you&apos;re picking up a barbell for the first time or working around a shoulder injury, FitSense AI adapts to you.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {personas.map((p) => (
            <div key={p.title} className="bg-white border border-slate-200 rounded-2xl p-7 shadow-sm border-l-4 border-l-blue-500 hover:shadow-md transition-shadow">
              <div className="text-4xl mb-3">{p.icon}</div>
              <h3 className="text-lg font-semibold text-slate-900 mb-1" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>{p.title}</h3>
              <p className="text-blue-600 text-sm font-medium mb-3">{p.subtitle}</p>
              <p className="text-slate-600 text-sm leading-relaxed mb-5">{p.desc}</p>
              <div className="flex flex-wrap gap-2">
                {p.tags.map((tag) => (
                  <span key={tag} className="px-3 py-1 bg-blue-50 text-blue-700 text-xs font-medium rounded-full">{tag}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
