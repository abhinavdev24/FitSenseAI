import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'FitSense AI — Your AI Fitness Coach',
  description: 'Personalized workout plans that adapt week-over-week, built on a fine-tuned AI model trained on real coaching expertise.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet" />
      </head>
      <body className={`${inter.className} bg-slate-50 text-slate-900 antialiased font-sans`}>
        {children}
      </body>
    </html>
  )
}
