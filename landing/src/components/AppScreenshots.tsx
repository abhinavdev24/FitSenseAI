const screens = [
  {
    tab: 'Home',
    icon: '🏠',
    headline: 'Dashboard',
    lines: ['Goal · days/week · experience', 'Avg sleep, calories, weight', 'Current plan summary', 'Recent workout history'],
    accent: 'from-blue-500 to-blue-700',
  },
  {
    tab: 'Plan',
    icon: '📅',
    headline: 'Your Plan',
    lines: ['AI-generated weekly structure', 'Sets · reps · RIR per exercise', 'Modify in plain English', 'Adapts from your logs'],
    accent: 'from-indigo-500 to-blue-600',
    featured: true,
  },
  {
    tab: 'Workout',
    icon: '🏋️',
    headline: 'Log Session',
    lines: ['Pick today\'s plan day', 'Log reps · weight · RIR per set', 'Saved instantly to backend', 'Feeds next-week adaptation'],
    accent: 'from-blue-600 to-cyan-500',
  },
  {
    tab: 'Check-in',
    icon: '📊',
    headline: 'Daily Check-in',
    lines: ['Sleep hours', 'Calorie intake', 'Body weight', 'All feed the AI engine'],
    accent: 'from-sky-500 to-blue-500',
  },
  {
    tab: 'Coach',
    icon: '💬',
    headline: 'AI Coach',
    lines: ['Ask anything about training', 'Knows your plan & history', 'Safety-aware responses', 'Suggests plan modifications'],
    accent: 'from-blue-700 to-indigo-600',
  },
]

export default function AppScreenshots() {
  return (
    <section className="py-24 px-6 bg-gradient-to-b from-blue-50 to-white">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-blue-900 mb-4" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            Five tabs. Everything you need.
          </h2>
          <p className="text-slate-600 text-lg max-w-xl mx-auto">
            A focused interface built for real training — no clutter, no upsells.
          </p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 items-end">
          {screens.map((s) => (
            <div
              key={s.tab}
              className={`flex flex-col rounded-3xl overflow-hidden shadow-lg border-4 border-white transition-all duration-300 hover:-translate-y-1 hover:shadow-xl ${s.featured ? 'lg:-translate-y-4 shadow-2xl' : ''}`}
            >
              {/* Phone top notch */}
              <div className={`bg-gradient-to-br ${s.accent} px-4 pt-5 pb-6`}>
                <div className="w-10 h-1.5 bg-white/30 rounded-full mx-auto mb-4" />
                <div className="text-3xl mb-2 text-center">{s.icon}</div>
                <h3 className="text-white font-bold text-center text-sm" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                  {s.headline}
                </h3>
              </div>
              {/* Screen body */}
              <div className="bg-[#0f1117] px-3 py-4 flex-1 space-y-2">
                {s.lines.map((line) => (
                  <div key={line} className="flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-400 shrink-0" />
                    <span className="text-slate-300 text-xs leading-snug">{line}</span>
                  </div>
                ))}
              </div>
              {/* Tab bar label */}
              <div className="bg-[#0f1117] border-t border-white/10 py-2 text-center">
                <span className="text-blue-400 text-xs font-semibold">{s.tab}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
