import 'package:flutter/material.dart';

import '../../app_theme.dart';
import '../../services/api_service.dart';
import '../../services/session_store.dart';
import '../home/home_shell.dart';

class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key, required this.session});
  final SessionStore session;

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  int _step = 0;
  bool _loading = false;
  String? _error;

  // Step 1 — goal
  String _goal = 'general_fitness';
  // Step 2 — experience
  String _experience = 'beginner';
  // Step 3 — schedule
  int _daysPerWeek = 3;
  String _equipment = 'full_gym';
  // Step 4 — constraints
  final _constraintsController = TextEditingController();

  Future<void> _finish() async {
    setState(() { _loading = true; _error = null; });
    try {
      final api = ApiService(token: widget.session.token);
      await api.saveOnboarding({
        'goal_type': _goal,
        'experience_level': _experience,
        'days_per_week': _daysPerWeek,
        'equipment': _equipment,
        'constraints': _constraintsController.text.trim(),
      });
      final planRes = await api.createPlan({
        'goal_type': _goal,
        'experience_level': _experience,
        'days_per_week': _daysPerWeek,
        'equipment': _equipment,
        'constraints': _constraintsController.text.trim(),
      });
      final jobId = planRes['job_id'] as String?;
      if (jobId != null) await widget.session.setPendingPlanJobId(jobId);
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => HomeShell(session: widget.session, initialIndex: 1)),
      );
    } catch (e) {
      setState(() => _error = e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Setup ${_step + 1}/4'),
        automaticallyImplyLeading: _step > 0,
        leading: _step > 0 ? IconButton(icon: const Icon(Icons.arrow_back), onPressed: () => setState(() => _step--)) : null,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 480),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (_step == 0) ..._goalStep(),
                if (_step == 1) ..._experienceStep(),
                if (_step == 2) ..._scheduleStep(),
                if (_step == 3) ..._constraintsStep(),
                if (_error != null) ...[
                  const SizedBox(height: 12),
                  Text(_error!, style: const TextStyle(color: Colors.redAccent)),
                ],
                const SizedBox(height: 24),
                ElevatedButton(
                  onPressed: _loading ? null : () {
                    if (_step < 3) {
                      setState(() => _step++);
                    } else {
                      _finish();
                    }
                  },
                  child: Text(_loading ? 'Creating your plan...' : _step < 3 ? 'Next →' : 'Generate My Plan'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  List<Widget> _goalStep() => [
    const Text('What is your main goal?', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
    const SizedBox(height: 20),
    ...[
      ('fat_loss', 'Fat Loss', 'Lose body fat while preserving muscle'),
      ('muscle_gain', 'Muscle Gain', 'Build size and strength'),
      ('strength', 'Strength', 'Increase your lifts'),
      ('general_fitness', 'General Fitness', 'Stay healthy and active'),
    ].map((e) => _choice(e.$1, e.$2, e.$3, _goal, (v) => setState(() => _goal = v))),
  ];

  List<Widget> _experienceStep() => [
    const Text('Experience level?', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
    const SizedBox(height: 20),
    ...[
      ('beginner', 'Beginner', 'Less than 1 year of consistent training'),
      ('intermediate', 'Intermediate', '1–3 years of training'),
      ('advanced', 'Advanced', '3+ years of structured programming'),
    ].map((e) => _choice(e.$1, e.$2, e.$3, _experience, (v) => setState(() => _experience = v))),
  ];

  List<Widget> _scheduleStep() => [
    const Text('Training schedule', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
    const SizedBox(height: 20),
    const Text('Days per week', style: TextStyle(color: AppTheme.muted)),
    const SizedBox(height: 8),
    Row(
      children: List.generate(6, (i) {
        final d = i + 2;
        return Padding(
          padding: const EdgeInsets.only(right: 8),
          child: ChoiceChip(
            label: Text('$d'),
            selected: _daysPerWeek == d,
            onSelected: (_) => setState(() => _daysPerWeek = d),
            selectedColor: AppTheme.primary,
            labelStyle: TextStyle(color: _daysPerWeek == d ? Colors.black : Colors.white),
          ),
        );
      }),
    ),
    const SizedBox(height: 20),
    const Text('Available equipment', style: TextStyle(color: AppTheme.muted)),
    const SizedBox(height: 8),
    ...[
      ('full_gym', 'Full Gym', 'Barbells, machines, cables'),
      ('dumbbells', 'Dumbbells Only', 'Dumbbells and bench'),
      ('bodyweight', 'Bodyweight', 'No equipment'),
      ('home_gym', 'Home Gym', 'Basic home setup'),
    ].map((e) => _choice(e.$1, e.$2, e.$3, _equipment, (v) => setState(() => _equipment = v))),
  ];

  List<Widget> _constraintsStep() => [
    const Text('Any injuries or constraints?', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
    const SizedBox(height: 8),
    const Text('Optional — describe any injuries, pain points, or movements to avoid.', style: TextStyle(color: AppTheme.muted)),
    const SizedBox(height: 20),
    TextField(
      controller: _constraintsController,
      maxLines: 4,
      decoration: const InputDecoration(
        labelText: 'e.g. left knee pain, avoid overhead pressing',
        alignLabelWithHint: true,
      ),
    ),
  ];

  Widget _choice(String value, String label, String desc, String current, ValueChanged<String> onTap) {
    final selected = current == value;
    return GestureDetector(
      onTap: () => onTap(value),
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: selected ? AppTheme.primary.withValues(alpha: 0.12) : AppTheme.surface,
          border: Border.all(color: selected ? AppTheme.primary : AppTheme.border),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Row(
          children: [
            Icon(selected ? Icons.check_circle : Icons.circle_outlined, color: selected ? AppTheme.primary : AppTheme.muted, size: 20),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(label, style: TextStyle(fontWeight: FontWeight.w600, color: selected ? AppTheme.primary : Colors.white)),
                  Text(desc, style: const TextStyle(color: AppTheme.muted, fontSize: 12)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
