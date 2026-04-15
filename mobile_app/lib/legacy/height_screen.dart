import 'package:flutter/material.dart';
import 'goal_screen.dart';

class HeightScreen extends StatefulWidget {
  const HeightScreen({super.key});

  @override
  State<HeightScreen> createState() => _HeightScreenState();
}

class _HeightScreenState extends State<HeightScreen> {
  String enteredHeight = '';
  bool isCm = true;

  void onKeyPress(String value) {
    setState(() {
      if (enteredHeight.length < 3) {
        enteredHeight += value;
      }
    });
  }

  void onDelete() {
    setState(() {
      if (enteredHeight.isNotEmpty) {
        enteredHeight = enteredHeight.substring(0, enteredHeight.length - 1);
      }
    });
  }

  void onEnter() {
    if (enteredHeight.isNotEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Height: $enteredHeight ${isCm ? 'cm' : 'ft'} selected',
          ),
          backgroundColor: const Color(0xFF5B7FFF),
        ),
      );
    }
  }

  Widget buildKey(String value) {
    return GestureDetector(
      onTap: () => onKeyPress(value),
      child: Container(
        decoration: BoxDecoration(
          color: const Color(0xFF1C1E2A),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Center(
          child: Text(
            value,
            style: const TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: Color(0xFFEAEDF5),
            ),
          ),
        ),
      ),
    );
  }

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
              const SizedBox(height: 30),
              const Text(
                'What Is Your Height?',
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFFEAEDF5),
                ),
              ),
              const SizedBox(height: 8),
              const Text(
                'We use this to personalize your workouts',
                style: TextStyle(
                  fontSize: 14,
                  color: Color(0xFF8B92A5),
                ),
              ),
              const SizedBox(height: 20),
              // CM / FT toggle
              Container(
                decoration: BoxDecoration(
                  color: const Color(0xFF1C1E2A),
                  borderRadius: BorderRadius.circular(30),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: GestureDetector(
                        onTap: () => setState(() {
                          isCm = true;
                          enteredHeight = '';
                        }),
                        child: Container(
                          padding: const EdgeInsets.symmetric(vertical: 12),
                          decoration: BoxDecoration(
                            color: isCm
                                ? const Color(0xFF5B7FFF)
                                : Colors.transparent,
                            borderRadius: BorderRadius.circular(30),
                          ),
                          child: Center(
                            child: Text(
                              'CM',
                              style: TextStyle(
                                color: isCm
                                    ? Colors.white
                                    : const Color(0xFF8B92A5),
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                    Expanded(
                      child: GestureDetector(
                        onTap: () => setState(() {
                          isCm = false;
                          enteredHeight = '';
                        }),
                        child: Container(
                          padding: const EdgeInsets.symmetric(vertical: 12),
                          decoration: BoxDecoration(
                            color: !isCm
                                ? const Color(0xFF5B7FFF)
                                : Colors.transparent,
                            borderRadius: BorderRadius.circular(30),
                          ),
                          child: Center(
                            child: Text(
                              'FT',
                              style: TextStyle(
                                color: !isCm
                                    ? Colors.white
                                    : const Color(0xFF8B92A5),
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              // Height display
              Expanded(
                child: Center(
                  child: RichText(
                    text: TextSpan(
                      children: [
                        TextSpan(
                          text: enteredHeight.isEmpty ? '_ _ _' : enteredHeight,
                          style: TextStyle(
                            fontSize: 80,
                            fontWeight: FontWeight.bold,
                            color: enteredHeight.isEmpty
                                ? const Color(0xFF8B92A5)
                                : const Color(0xFFEAEDF5),
                          ),
                        ),
                        TextSpan(
                          text: isCm ? ' cm' : ' ft',
                          style: const TextStyle(
                            fontSize: 24,
                            color: Color(0xFF8B92A5),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              // Keypad
              GridView.count(
                shrinkWrap: true,
                crossAxisCount: 3,
                mainAxisSpacing: 12,
                crossAxisSpacing: 12,
                childAspectRatio: 1.5,
                physics: const NeverScrollableScrollPhysics(),
                children: [
                  buildKey('1'),
                  buildKey('2'),
                  buildKey('3'),
                  buildKey('4'),
                  buildKey('5'),
                  buildKey('6'),
                  buildKey('7'),
                  buildKey('8'),
                  buildKey('9'),
                  GestureDetector(
                    onTap: onDelete,
                    child: Container(
                      decoration: BoxDecoration(
                        color: const Color(0xFF1C1E2A),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: const Center(
                        child: Icon(
                          Icons.backspace_outlined,
                          color: Color(0xFF5B7FFF),
                          size: 24,
                        ),
                      ),
                    ),
                  ),
                  buildKey('0'),
                  GestureDetector(
                    onTap: onEnter,
                    child: Container(
                      decoration: BoxDecoration(
                        color: const Color(0xFF4CAF8C),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: const Center(
                        child: Icon(
                          Icons.check,
                          color: Colors.white,
                          size: 24,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: enteredHeight.isEmpty
                    ? null
                    : () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => const GoalScreen(),
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