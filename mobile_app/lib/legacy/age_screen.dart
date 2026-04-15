import 'package:flutter/material.dart';
import 'weight_screen.dart';

class AgeScreen extends StatefulWidget {
  const AgeScreen({super.key});

  @override
  State<AgeScreen> createState() => _AgeScreenState();
}

class _AgeScreenState extends State<AgeScreen> {
  String enteredAge = '';

  void onKeyPress(String value) {
    setState(() {
      if (enteredAge.length < 2) {
        enteredAge += value;
      }
    });
  }

  void onDelete() {
    setState(() {
      if (enteredAge.isNotEmpty) {
        enteredAge = enteredAge.substring(0, enteredAge.length - 1);
      }
    });
  }

  void onEnter() {
    if (enteredAge.isNotEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Age: $enteredAge selected'),
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
                'How Old Are You?',
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFFEAEDF5),
                ),
              ),
              const SizedBox(height: 8),
              const Text(
                'We use this to calculate your fitness plan',
                style: TextStyle(
                  fontSize: 14,
                  color: Color(0xFF8B92A5),
                ),
              ),
              const SizedBox(height: 20),
              // Age display
              Expanded(
                child: Center(
                  child: RichText(
                    text: TextSpan(
                      children: [
                        TextSpan(
                          text: enteredAge.isEmpty ? '_ _' : enteredAge,
                          style: TextStyle(
                            fontSize: 80,
                            fontWeight: FontWeight.bold,
                            color: enteredAge.isEmpty
                                ? const Color(0xFF8B92A5)
                                : const Color(0xFFEAEDF5),
                          ),
                        ),
                        const TextSpan(
                          text: ' yrs',
                          style: TextStyle(
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
                onPressed: enteredAge.isEmpty
                    ? null
                    : () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => const WeightScreen(),
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