import 'package:flutter/material.dart';

import '../../services/api_service.dart';
import '../../services/session_store.dart';
import '../../widgets/common.dart';

class CheckinTab extends StatefulWidget {
  const CheckinTab({super.key, required this.session});
  final SessionStore session;

  @override
  State<CheckinTab> createState() => _CheckinTabState();
}

class _CheckinTabState extends State<CheckinTab> {
  final _sleep = TextEditingController();
  final _calories = TextEditingController();
  final _weight = TextEditingController();
  String? _message;

  Future<void> _submit(String type) async {
    setState(() => _message = 'Saving $type...');
    try {
      final api = ApiService(token: widget.session.token);
      final now = DateTime.now();
      if (type == 'sleep') {
        await api.logSleep(now, double.parse(_sleep.text));
      } else if (type == 'calories') {
        await api.logCalories(now, int.parse(_calories.text));
      } else {
        await api.logWeight(now, double.parse(_weight.text));
      }
      setState(() => _message = '${type[0].toUpperCase()}${type.substring(1)} log saved.');
    } catch (e) {
      setState(() => _message = e.toString().replaceFirst('Exception: ', ''));
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        SectionCard(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Daily check-in', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 20)),
              const SizedBox(height: 8),
              const Text('Log just the essentials the backend expects: sleep, calories, and body weight.', style: TextStyle(color: Colors.white70)),
              const SizedBox(height: 16),
              TextField(controller: _sleep, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Sleep hours')), const SizedBox(height: 10),
              OutlinedButton(onPressed: () => _submit('sleep'), child: const Text('Save sleep')),
              const Divider(height: 28),
              TextField(controller: _calories, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Calories')), const SizedBox(height: 10),
              OutlinedButton(onPressed: () => _submit('calories'), child: const Text('Save calories')),
              const Divider(height: 28),
              TextField(controller: _weight, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Weight (kg)')), const SizedBox(height: 10),
              OutlinedButton(onPressed: () => _submit('weight'), child: const Text('Save weight')),
              if (_message != null) ...[
                const SizedBox(height: 12),
                Text(_message!, style: const TextStyle(color: Colors.white70)),
              ],
            ],
          ),
        ),
      ],
    );
  }
}
