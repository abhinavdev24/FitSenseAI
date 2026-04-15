import 'package:flutter/material.dart';
import 'age_screen.dart';

class GenderScreen extends StatefulWidget {
  const GenderScreen({super.key});

  @override
  State<GenderScreen> createState() => _GenderScreenState();
}

class _GenderScreenState extends State<GenderScreen> {
  String selectedGender = '';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF12131A),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 20),
              GestureDetector(
                onTap: () => Navigator.pop(context),
                child: const Icon(
                  Icons.arrow_back_ios,
                  color: Color(0xFF5B7FFF),
                ),
              ),
              const SizedBox(height: 40),
              const Text(
                "What's Your Gender?",
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFFEAEDF5),
                ),
              ),
              const SizedBox(height: 8),
              const Text(
                'This helps us personalize your plan',
                style: TextStyle(
                  fontSize: 14,
                  color: Color(0xFF8B92A5),
                ),
              ),
              const SizedBox(height: 60),
              // Male option
              GestureDetector(
                onTap: () => setState(() => selectedGender = 'Male'),
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    color: selectedGender == 'Male'
                        ? const Color(0xFF5B7FFF)
                        : const Color(0xFF1C1E2A),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: selectedGender == 'Male'
                          ? const Color(0xFF5B7FFF)
                          : Colors.transparent,
                    ),
                  ),
                  child: Column(
                    children: [
                      Icon(
                        Icons.male,
                        size: 60,
                        color: selectedGender == 'Male'
                            ? Colors.white
                            : const Color(0xFF8B92A5),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'Male',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: selectedGender == 'Male'
                              ? Colors.white
                              : const Color(0xFFEAEDF5),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 20),
              // Female option
              GestureDetector(
                onTap: () => setState(() => selectedGender = 'Female'),
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    color: selectedGender == 'Female'
                        ? const Color(0xFF5B7FFF)
                        : const Color(0xFF1C1E2A),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: selectedGender == 'Female'
                          ? const Color(0xFF5B7FFF)
                          : Colors.transparent,
                    ),
                  ),
                  child: Column(
                    children: [
                      Icon(
                        Icons.female,
                        size: 60,
                        color: selectedGender == 'Female'
                            ? Colors.white
                            : const Color(0xFF8B92A5),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'Female',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: selectedGender == 'Female'
                              ? Colors.white
                              : const Color(0xFFEAEDF5),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const Spacer(),
              ElevatedButton(
                onPressed: selectedGender.isEmpty
                    ? null
                    : () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => const AgeScreen(),
                          ),
                        );
                      },
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF5B7FFF),
                  disabledBackgroundColor: const Color(0xFF1C1E2A),
                  minimumSize: const Size(double.infinity, 55),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                ),
                child: const Text(
                  'Continue',
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              const SizedBox(height: 30),
            ],
          ),
        ),
      ),
    );
  }
}