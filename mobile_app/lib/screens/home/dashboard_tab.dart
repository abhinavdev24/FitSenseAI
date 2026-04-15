import 'package:flutter/material.dart';

import '../../services/api_service.dart';
import '../../services/session_store.dart';
import '../../widgets/common.dart';

class DashboardTab extends StatefulWidget {
  const DashboardTab({super.key, required this.session});
  final SessionStore session;

  @override
  State<DashboardTab> createState() => _DashboardTabState();
}

class _DashboardTabState extends State<DashboardTab> {
  Map<String, dynamic>? data;
  bool loading = true;
  String? error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      loading = true;
      error = null;
    });
    try {
      final api = ApiService(token: widget.session.token);
      final result = await api.dashboard();
      if (!mounted) return;
      setState(() => data = result);
    } catch (e) {
      setState(() => error = e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (loading) return const Center(child: CircularProgressIndicator());
    if (error != null) return Center(child: Text(error!));
    final profile = (data?['profile'] as Map<String, dynamic>? ?? {});
    final logs = (data?['recent_logs'] as Map<String, dynamic>? ?? {});
    final currentPlan = data?['current_plan'] as Map<String, dynamic>?;
    final activeJob = profile['active_plan_job'] as Map<String, dynamic>?;
    final workouts = (data?['recent_workouts'] as List?)?.cast<Map<String, dynamic>>() ?? [];
    final processing = activeJob != null && (activeJob['status'] == 'queued' || activeJob['status'] == 'running');
    return RefreshIndicator(
      onRefresh: _load,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          SectionCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Welcome back, ${widget.session.name ?? 'Athlete'}', style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
                const SizedBox(height: 6),
                Text('Goal: ${profile['goal'] ?? 'Not set'} • ${profile['days_per_week'] ?? '-'} days/week • ${profile['experience_level'] ?? '-'}', style: const TextStyle(color: Colors.white70)),
              ],
            ),
          ),
          if (processing)
            SectionCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Recommendation pipeline', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  Text(activeJob['progress_message']?.toString() ?? 'Generating your latest plan...'),
                  const SizedBox(height: 6),
                  const Text('The backend is processing in the background. You can keep using the app while the latest plan is prepared.', style: TextStyle(color: Colors.white70)),
                ],
              ),
            ),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              StatPill(label: 'Avg sleep', value: '${logs['avg_sleep_hours'] ?? '-'} h'),
              StatPill(label: 'Avg calories', value: '${logs['avg_calories'] ?? '-'} kcal'),
              StatPill(label: 'Latest weight', value: '${logs['latest_weight'] ?? '-'} kg'),
            ],
          ),
          SectionCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Current plan', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                Text(currentPlan?['name']?.toString() ?? 'No active plan'),
                const SizedBox(height: 6),
                Text(currentPlan?['explanation']?.toString() ?? 'Create a plan from onboarding or from the Plan tab.', style: const TextStyle(color: Colors.white70)),
              ],
            ),
          ),
          SectionCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Recent workouts', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                if (workouts.isEmpty)
                  const Text('No workouts logged yet.')
                else
                  ...workouts.take(3).map((w) => ListTile(
                        contentPadding: EdgeInsets.zero,
                        title: Text(w['started_at']?.toString().split('T').first ?? 'Workout'),
                        subtitle: Text('${w['exercise_count']} exercises • ${w['set_count']} sets'),
                      )),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
