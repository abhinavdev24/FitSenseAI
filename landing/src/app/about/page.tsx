import Link from 'next/link'
import Navbar from '@/components/Navbar'
import Footer from '@/components/Footer'

const team = [
  {
    name: 'Manoj Harri Doss',
    role: 'ML & Backend Engineer',
    bio: 'Led model fine-tuning, backend API development, and system integration. Built the teacher-student distillation pipeline and Flutter mobile app.',
  },
  {
    name: 'Anjali',
    role: 'ML Engineer',
    bio: 'Focused on data pipeline design and model evaluation. Contributed to synthetic dataset generation and quality filtering for fine-tuning.',
  },
  {
    name: 'Hrishikesh',
    role: 'ML Engineer',
    bio: 'Worked on model training infrastructure and experiment tracking. Contributed to the adaptation engine and coaching response quality.',
  },
  {
    name: 'Bhoomi',
    role: 'Data & ML Engineer',
    bio: 'Built data preprocessing and validation workflows. Ensured training data quality and contributed to the evaluation framework.',
  },
  {
    name: 'Harini',
    role: 'ML Research Engineer',
    bio: 'Researched distillation techniques and benchmarked model performance. Contributed to safety-aware response design and constraint handling.',
  },
  {
    name: 'Abhinav',
    role: 'Backend & DevOps Engineer',
    bio: 'Managed cloud infrastructure, CI/CD pipelines, and API testing. Ensured reliable deployment and integration across the system.',
  },
]

export default function AboutPage() {
  return (
    <>
      <Navbar />
      <main className="bg-slate-50 min-h-screen">
        <div className="max-w-5xl mx-auto px-6 py-20">
          <div className="text-center mb-16">
            <Link href="/" className="text-blue-600 hover:text-blue-700 text-sm font-medium mb-6 inline-block">← Back to Home</Link>
            <h1 className="text-4xl sm:text-5xl font-bold text-blue-900 mb-4" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
              Meet the Team
            </h1>
            <p className="text-slate-600 text-lg max-w-2xl mx-auto">
              Six graduate students who built FitSense AI — a full-stack AI fitness coaching system from model training to mobile app.
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {team.map((member) => (
              <div key={member.name} className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm hover:shadow-md transition-shadow">
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center text-white text-2xl font-bold mb-4">
                  {member.name[0]}
                </div>
                <h3 className="text-lg font-semibold text-slate-900 mb-1" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                  {member.name}
                </h3>
                <p className="text-blue-600 text-sm font-medium mb-1">{member.role}</p>
                <p className="text-xs text-slate-400 font-medium mb-3 uppercase tracking-wide">Graduate Student</p>
                <p className="text-slate-600 text-sm leading-relaxed">{member.bio}</p>
              </div>
            ))}
          </div>
        </div>
      </main>
      <Footer />
    </>
  )
}
