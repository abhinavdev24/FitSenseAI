import 'package:flutter/material.dart';

class GoalScreen extends StatefulWidget {
  const GoalScreen({super.key});

  @override
  State<GoalScreen> createState() => _GoalScreenState();
}

class _GoalScreenState extends State<GoalScreen> {
  String selectedGoal = '';

  final List<Map<String, dynamic>> goals = [
    {
      'title': 'Lose Weight',
      'icon': Icons.monitor_weight_outlined,
      'description': 'Burn fat and get lean',
    },
    {
      'title': 'Build Muscle',
      'icon': Icons.fitness_center,
      'description': 'Gain strength and mass',
    },
    {
      'title': 'Muscle Mass Gain',
      'icon': Icons.accessibility_new,
      'description': 'Increase muscle size',
    },
    {
      'title': 'Shape Body',
      'icon': Icons.self_improvement,
      'description': 'Tone and define your body',
    },
    {
      'title': 'Stay Active',
      'icon': Icons.directions_run,
      'description': 'Maintain a healthy lifestyle',
    },
  ];

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
                'What Is Your Goal?',
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFFEAEDF5),
                ),
              ),
              const SizedBox(height: 8),
              const Text(
                'We will build your plan around your goal',
                style: TextStyle(
                  fontSize: 14,
                  color: Color(0xFF8B92A5),
                ),
              ),
              const SizedBox(height: 30),
              Expanded(
                child: ListView.separated(
                  itemCount: goals.length,
                  separatorBuilder: (context, index) =>
                      const SizedBox(height: 12),
                  itemBuilder: (context, index) {
                    final goal = goals[index];
                    final isSelected = selectedGoal == goal['title'];
                    return GestureDetector(
                      onTap: () =>
                          setState(() => selectedGoal = goal['title']),
                      child: Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: isSelected
                              ? const Color(0xFF5B7FFF)
                              : const Color(0xFF1C1E2A),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                            color: isSelected
                                ? const Color(0xFF5B7FFF)
                                : Colors.transparent,
                          ),
                        ),
                        child: Row(
                          children: [
                            Container(
                              padding: const EdgeInsets.all(10),
                              decoration: BoxDecoration(
                                color: isSelected
                                    ? Colors.white.withOpacity(0.2)
                                    : const Color(0xFF12131A),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: Icon(
                                goal['icon'],
                                color: isSelected
                                    ? Colors.white
                                    : const Color(0xFF5B7FFF),
                                size: 24,
                              ),
                            ),
                            const SizedBox(width: 16),
                            Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  goal['title'],
                                  style: TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.bold,
                                    color: isSelected
                                        ? Colors.white
                                        : const Color(0xFFEAEDF5),
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  goal['description'],
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: isSelected
                                        ? Colors.white.withOpacity(0.8)
                                        : const Color(0xFF8B92A5),
                                  ),
                                ),
                              ],
                            ),
                            const Spacer(),
                            if (isSelected)
                              const Icon(
                                Icons.check_circle,
                                color: Colors.white,
                                size: 24,
                              ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: selectedGoal.isEmpty
                    ? null
                    : () {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text('Goal: $selectedGoal selected'),
                            backgroundColor: const Color(0xFF5B7FFF),
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