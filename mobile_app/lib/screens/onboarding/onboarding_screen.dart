import 'package:flutter/material.dart';

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
  final _age = TextEditingController(text: '24');
  final _height = TextEditingController(text: '175');
  final _weight = TextEditingController(text: '72');
  final _injuries = TextEditingController();
  final _conditions = TextEditingController();
  final _medications = TextEditingController();
  final _allergies = TextEditingController();
  final _calories = TextEditingController(text: '2200');
  final _sleep = TextEditingController(text: '8');

  String _sex = 'M';
  String _goal = 'fat loss';
  int _daysPerWeek = 4;
  String _experience = 'beginner';
  String _activity = 'lightly_active';
  final Set<String> _equipment = {'bodyweight', 'dumbbells'};
  bool _saving = false;
  String? _error;
  String? _status;

  Future<void> _submit() async {
    setState(() {
      _saving = true;
      _error = null;
      _status = 'Saving your profile...';
    });
    try {
      final api = ApiService(token: widget.session.token);
      await api.saveOnboarding({
        'age': int.parse(_age.text),
        'sex': _sex,
        'height_cm': double.parse(_height.text),
        'weight_kg': double.parse(_weight.text),
        'goal_name': _goal,
        'days_per_week': _daysPerWeek,
        'experience_level': _experience,
        'activity_level': _activity,
        'equipment': _equipment.toList(),
        'injuries': _injuries.text.trim(),
        'conditions': _splitCsv(_conditions.text),
        'medications': _splitCsv(_medications.text),
        'allergies': _splitCsv(_allergies.text),
        'calorie_target': int.tryParse(_calories.text),
        'sleep_target_hours': double.tryParse(_sleep.text),
      });
      setState(() => _status = 'Profile saved. Queueing your personalized plan...');
      final job = await api.createPlan({
        'goal_name': _goal,
        'days_per_week': _daysPerWeek,
        'equipment': _equipment.toList(),
        'experience_level': _experience,
      });
      await widget.session.savePendingPlanJob(
        jobId: job['job_id'] as String,
        jobType: job['job_type']?.toString() ?? 'generate',
      );
      if (!mounted) return;
      Navigator.of(context).pushAndRemoveUntil(
        MaterialPageRoute(builder: (_) => HomeShell(session: widget.session, initialIndex: 1)),
        (route) => false,
      );
    } catch (e) {
      setState(() => _error = e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  List<String> _splitCsv(String raw) => raw.split(',').map((e) => e.trim()).where((e) => e.isNotEmpty).toList();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Set up your profile')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 700),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'This screen submits onboarding data quickly, queues the recommendation pipeline on the backend, and lets the app keep showing loading or the latest known plan while generation finishes in the background.',
                  style: TextStyle(color: Colors.white70),
                ),
                const SizedBox(height: 20),
                Wrap(
                  spacing: 14,
                  runSpacing: 14,
                  children: [
                    _textField(_age, 'Age'),
                    _textField(_height, 'Height (cm)'),
                    _textField(_weight, 'Weight (kg)'),
                    _textField(_calories, 'Daily calorie target'),
                    _textField(_sleep, 'Sleep target (hours)'),
                  ],
                ),
                const SizedBox(height: 14),
                DropdownButtonFormField<String>(
                  initialValue: _sex,
                  items: const ['M', 'F', 'other'].map((e) => DropdownMenuItem(value: e, child: Text(e))).toList(),
                  onChanged: (v) => setState(() => _sex = v!),
                  decoration: const InputDecoration(labelText: 'Sex'),
                ),
                const SizedBox(height: 14),
                DropdownButtonFormField<String>(
                  initialValue: _goal,
                  items: const ['fat loss', 'muscle gain', 'strength', 'general fitness']
                      .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                      .toList(),
                  onChanged: (v) => setState(() => _goal = v!),
                  decoration: const InputDecoration(labelText: 'Primary goal'),
                ),
                const SizedBox(height: 14),
                DropdownButtonFormField<int>(
                  initialValue: _daysPerWeek,
                  items: [2, 3, 4, 5, 6].map((e) => DropdownMenuItem(value: e, child: Text('$e days / week'))).toList(),
                  onChanged: (v) => setState(() => _daysPerWeek = v!),
                  decoration: const InputDecoration(labelText: 'Training frequency'),
                ),
                const SizedBox(height: 14),
                DropdownButtonFormField<String>(
                  initialValue: _experience,
                  items: const ['beginner', 'intermediate', 'advanced']
                      .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                      .toList(),
                  onChanged: (v) => setState(() => _experience = v!),
                  decoration: const InputDecoration(labelText: 'Experience level'),
                ),
                const SizedBox(height: 14),
                DropdownButtonFormField<String>(
                  initialValue: _activity,
                  items: const ['sedentary', 'lightly_active', 'moderately_active', 'very_active']
                      .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                      .toList(),
                  onChanged: (v) => setState(() => _activity = v!),
                  decoration: const InputDecoration(labelText: 'Daily activity level'),
                ),
                const SizedBox(height: 18),
                const Text('Equipment you have', style: TextStyle(fontWeight: FontWeight.w600)),
                Wrap(
                  spacing: 8,
                  children: ['bodyweight', 'dumbbells', 'barbell', 'machines', 'cables']
                      .map((item) => FilterChip(
                            label: Text(item),
                            selected: _equipment.contains(item),
                            onSelected: (selected) {
                              setState(() {
                                if (selected) {
                                  _equipment.add(item);
                                } else {
                                  _equipment.remove(item);
                                }
                              });
                            },
                          ))
                      .toList(),
                ),
                const SizedBox(height: 18),
                TextField(controller: _injuries, decoration: const InputDecoration(labelText: 'Injuries / pain notes')),
                const SizedBox(height: 14),
                TextField(controller: _conditions, decoration: const InputDecoration(labelText: 'Conditions (comma-separated)')),
                const SizedBox(height: 14),
                TextField(controller: _medications, decoration: const InputDecoration(labelText: 'Medications (comma-separated)')),
                const SizedBox(height: 14),
                TextField(controller: _allergies, decoration: const InputDecoration(labelText: 'Allergies (comma-separated)')),
                if (_status != null) ...[
                  const SizedBox(height: 12),
                  Text(_status!, style: const TextStyle(color: Colors.white70)),
                ],
                if (_error != null) ...[
                  const SizedBox(height: 12),
                  Text(_error!, style: const TextStyle(color: Colors.redAccent)),
                ],
                const SizedBox(height: 22),
                ElevatedButton(
                  onPressed: _saving ? null : _submit,
                  child: Text(_saving ? 'Submitting...' : 'Finish setup'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _textField(TextEditingController controller, String label) {
    return SizedBox(
      width: 220,
      child: TextField(controller: controller, keyboardType: TextInputType.number, decoration: InputDecoration(labelText: label)),
    );
  }
}
