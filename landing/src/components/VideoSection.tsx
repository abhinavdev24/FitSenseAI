export default function VideoSection() {
  return (
    <section className="py-24 px-6 bg-white">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl sm:text-4xl font-bold text-blue-900 mb-4" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            How the AI model works
          </h2>
          <p className="text-slate-600 text-lg max-w-2xl mx-auto">
            A deep dive into the teacher-student distillation pipeline that powers FitSense AI — from synthetic data generation to fine-tuned model deployment.
          </p>
        </div>
        <div className="relative w-full aspect-video rounded-2xl overflow-hidden shadow-2xl border border-slate-200">
          <iframe
            src="https://www.youtube.com/embed/CN3Cp9aeNxA"
            title="FitSense AI — Model Explanation"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
            className="absolute inset-0 w-full h-full"
          />
        </div>
      </div>
    </section>
  )
}
