export default function About() {
  return (
    <section id="about" className="py-24 px-6 bg-slate-50">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-blue-900 mb-4" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            About FitSense AI
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
          <div className="bg-white border border-slate-200 rounded-2xl p-8 shadow-sm">
            <h3 className="text-xl font-semibold text-blue-900 mb-4" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
              The Problem We Solve
            </h3>
            <p className="text-slate-600 leading-relaxed mb-4">
              People struggle to convert fitness goals into structured, safe, and adaptive weekly programs — especially when real-world constraints exist like injuries, limited equipment, or variable schedules.
            </p>
            <p className="text-slate-600 leading-relaxed mb-4">Generic plans and basic chatbots fail on the things that matter most:</p>
            <ul className="space-y-2">
              {[
                'Deep personalization for your constraints and history',
                'Consistent week-over-week adaptation from real performance data',
                'Explainable, actionable guidance — not just a list of exercises',
                'Safety-first responses for injury and medical risk scenarios',
              ].map((item) => (
                <li key={item} className="flex items-start gap-2 text-slate-600 text-sm">
                  <span className="text-blue-500 mt-0.5 shrink-0">✓</span>
                  {item}
                </li>
              ))}
            </ul>
          </div>
          <div className="bg-white border border-slate-200 rounded-2xl p-8 shadow-sm">
            <h3 className="text-xl font-semibold text-blue-900 mb-4" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
              How the AI Works
            </h3>
            <p className="text-slate-600 leading-relaxed mb-6">
              FitSense AI is powered by a custom fine-tuned model built using a teacher-student distillation pipeline:
            </p>
            <div className="space-y-4">
              {[
                { step: '1', title: 'Teacher LLM', desc: 'A large foundation model generates high-quality coaching outputs across thousands of realistic scenarios.' },
                { step: '2', title: 'Distillation', desc: 'Those outputs are filtered, validated, and used to train a compact student model — preserving coaching quality at a fraction of the cost.' },
                { step: '3', title: 'Student Model', desc: 'The fine-tuned model runs inference fast, handles your queries in seconds, and is purpose-built for fitness coaching.' },
              ].map(({ step, title, desc }) => (
                <div key={step} className="flex gap-4">
                  <div className="w-8 h-8 rounded-full bg-blue-600 text-white text-sm font-bold flex items-center justify-center shrink-0">
                    {step}
                  </div>
                  <div>
                    <p className="font-semibold text-slate-800 text-sm">{title}</p>
                    <p className="text-slate-500 text-sm leading-relaxed">{desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
