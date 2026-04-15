import 'package:flutter/material.dart';

import '../../services/api_service.dart';
import '../../services/session_store.dart';
import '../../widgets/common.dart';

class WorkoutTab extends StatefulWidget {
  const WorkoutTab({super.key, required this.session});
  final SessionStore session;

  @override
  State<WorkoutTab> createState() => _WorkoutTabState();
}

class _WorkoutTabState extends State<WorkoutTab> {
  Map<String, dynamic>? _plan;
  Map<String, dynamic>? _selectedDay;
  bool _loading = true;
  String? _message;

  final Map<String, TextEditingController> _reps = {};
  final Map<String, TextEditingController> _weights = {};
  final Map<String, TextEditingController> _rir = {};

  @override
  void initState() {
    super.initState();
    _loadPlan();
  }

  Future<void> _loadPlan() async {
    setState(() {
      _loading = true;
      _message = null;
    });
    try {
      final api = ApiService(token: widget.session.token);
      final data = await api.currentPlan();
      final plan = data['plan'] as Map<String, dynamic>?;
      final firstDay = (plan?['days'] as List?)?.cast<Map<String, dynamic>>().isNotEmpty == true
          ? ((plan!['days'] as List).first as Map<String, dynamic>)
          : null;
      if (!mounted) return;
      setState(() {
        _plan = plan;
        _selectedDay = firstDay;
        _seedControllers();
      });
    } catch (e) {
      setState(() => _message = e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _seedControllers() {
    _reps.clear();
    _weights.clear();
    _rir.clear();
    final exercises = (_selectedDay?['exercises'] as List?)?.cast<Map<String, dynamic>>() ?? [];
    for (final ex in exercises) {
      for (final set in (ex['sets'] as List?)?.cast<Map<String, dynamic>>() ?? []) {
        final key = '${ex['plan_exercise_id']}_${set['set_number']}';
        _reps[key] = TextEditingController(text: '${set['target_reps']}');
        _weights[key] = TextEditingController(text: '0');
        _rir[key] = TextEditingController(text: '${set['target_rir']}');
      }
    }
  }

  Future<void> _submitWorkout() async {
    if (_plan == null || _selectedDay == null) return;
    setState(() => _message = 'Saving workout...');
    try {
      final api = ApiService(token: widget.session.token);
      final workoutRes = await api.createWorkout({
        'plan_id': _plan!['plan_id'],
        'plan_day_id': _selectedDay!['plan_day_id'],
        'started_at': DateTime.now().toUtc().toIso8601String(),
      });
      final workoutId = workoutRes['workout_id'] as String;
      final exercises = (_selectedDay!['exercises'] as List?)?.cast<Map<String, dynamic>>() ?? [];
      for (final ex in exercises) {
        final exRes = await api.createWorkoutExercise(workoutId, {
          'exercise_id': ex['exercise_id'],
          'plan_exercise_id': ex['plan_exercise_id'],
          'position': ex['position'],
        });
        final workoutExerciseId = exRes['workout_exercise_id'] as String;
        for (final set in (ex['sets'] as List?)?.cast<Map<String, dynamic>>() ?? []) {
          final key = '${ex['plan_exercise_id']}_${set['set_number']}';
          await api.createWorkoutSet(workoutId, {
            'workout_exercise_id': workoutExerciseId,
            'set_number': set['set_number'],
            'reps': int.tryParse(_reps[key]?.text ?? '') ?? set['target_reps'],
            'weight': double.tryParse(_weights[key]?.text ?? '') ?? 0,
            'rir': int.tryParse(_rir[key]?.text ?? '') ?? set['target_rir'],
            'is_warmup': false,
            'completed_at': DateTime.now().toUtc().toIso8601String(),
          });
        }
      }
      setState(() => _message = 'Workout saved successfully.');
    } catch (e) {
      setState(() => _message = e.toString().replaceFirst('Exception: ', ''));
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) return const Center(child: CircularProgressIndicator());
    if (_plan == null) return Center(child: Text(_message ?? 'No active plan yet. Create one from onboarding or the plan tab.'));
    final days = (_plan!['days'] as List).cast<Map<String, dynamic>>();
    final exercises = (_selectedDay?['exercises'] as List?)?.cast<Map<String, dynamic>>() ?? [];
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        SectionCard(
          child: DropdownButtonFormField<String>(
            initialValue: _selectedDay?['plan_day_id'] as String?,
            decoration: const InputDecoration(labelText: 'Choose plan day'),
            items: days.map((day) => DropdownMenuItem(value: day['plan_day_id'] as String, child: Text(day['name'].toString()))).toList(),
            onChanged: (value) {
              final selected = days.firstWhere((day) => day['plan_day_id'] == value);
              setState(() {
                _selectedDay = selected;
                _seedControllers();
              });
            },
          ),
        ),
        ...exercises.map((ex) => SectionCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(ex['exercise_name'].toString(), style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 17)),
                  const SizedBox(height: 8),
                  ...((ex['sets'] as List?)?.cast<Map<String, dynamic>>() ?? []).map((set) {
                    final key = '${ex['plan_exercise_id']}_${set['set_number']}';
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: Row(
                        children: [
                          Expanded(child: Text('Set ${set['set_number']}')),
                          SizedBox(width: 80, child: TextField(controller: _reps[key], keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Reps'))),
                          const SizedBox(width: 8),
                          SizedBox(width: 84, child: TextField(controller: _weights[key], keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'kg'))),
                          const SizedBox(width: 8),
                          SizedBox(width: 74, child: TextField(controller: _rir[key], keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'RIR'))),
                        ],
                      ),
                    );
                  }),
                ],
              ),
            )),
        ElevatedButton(onPressed: _submitWorkout, child: const Text('Save workout session')),
        if (_message != null) ...[
          const SizedBox(height: 10),
          Text(_message!, style: const TextStyle(color: Colors.white70)),
        ],
      ],
    );
  }
}
