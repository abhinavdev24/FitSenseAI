import 'dart:async';

import 'package:flutter/material.dart';

import '../../services/api_service.dart';
import '../../services/session_store.dart';
import '../../widgets/common.dart';

class PlanTab extends StatefulWidget {
  const PlanTab({super.key, required this.session});
  final SessionStore session;

  @override
  State<PlanTab> createState() => _PlanTabState();
}

class _PlanTabState extends State<PlanTab> {
  Map<String, dynamic>? _plan;
  Map<String, dynamic>? _activeJob;
  bool _loading = true;
  bool _queueing = false;
  final _modifyController = TextEditingController();
  String? _message;
  Timer? _pollTimer;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _modifyController.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _message = null;
    });
    final api = ApiService(token: widget.session.token);
    try {
      final result = await api.currentPlan();
      final pendingJobId = widget.session.pendingPlanJobId;
      Map<String, dynamic>? job = result['active_job'] as Map<String, dynamic>?;
      if (job == null && pendingJobId != null && pendingJobId.isNotEmpty) {
        try {
          job = await api.planJobStatus(pendingJobId);
        } catch (_) {}
      }
      if (!mounted) return;
      setState(() {
        _plan = result['plan'] as Map<String, dynamic>?;
        _activeJob = job;
        _message = _jobMessage(job);
      });
      _syncPolling(job);
    } catch (e) {
      setState(() => _message = e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  String? _jobMessage(Map<String, dynamic>? job) {
    if (job == null) return null;
    final status = job['status']?.toString();
    if (status == 'failed') return job['error_message']?.toString() ?? 'Plan job failed.';
    return job['progress_message']?.toString() ?? 'Plan job is running in the background.';
  }

  void _syncPolling(Map<String, dynamic>? job) {
    _pollTimer?.cancel();
    if (job == null) return;
    final status = job['status']?.toString();
    if (status == 'queued' || status == 'running') {
      _pollTimer = Timer.periodic(const Duration(seconds: 2), (_) => _pollJob());
    }
  }

  Future<void> _pollJob() async {
    final jobId = widget.session.pendingPlanJobId ?? _activeJob?['job_id']?.toString();
    if (jobId == null || jobId.isEmpty) {
      _pollTimer?.cancel();
      return;
    }
    final api = ApiService(token: widget.session.token);
    try {
      final job = await api.planJobStatus(jobId);
      if (!mounted) return;
      final status = job['status']?.toString();
      if (status == 'completed') {
        await widget.session.clearPendingPlanJob();
        _pollTimer?.cancel();
        setState(() {
          _activeJob = null;
          _plan = job['result_plan'] as Map<String, dynamic>? ?? _plan;
          _message = job['progress_message']?.toString() ?? 'Plan ready.';
        });
      } else if (status == 'failed') {
        await widget.session.clearPendingPlanJob();
        _pollTimer?.cancel();
        setState(() {
          _activeJob = job;
          _message = job['error_message']?.toString() ?? 'Plan job failed.';
        });
      } else {
        setState(() {
          _activeJob = job;
          _plan = job['latest_plan'] as Map<String, dynamic>? ?? _plan;
          _message = _jobMessage(job);
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _message = e.toString().replaceFirst('Exception: ', ''));
    }
  }

  Future<void> _generate() async {
    final api = ApiService(token: widget.session.token);
    setState(() {
      _queueing = true;
      _message = 'Queueing plan generation...';
    });
    try {
      final result = await api.triggerPipeline({});
      await widget.session.savePendingPlanJob(
        jobId: result['job_id'] as String,
        jobType: result['job_type']?.toString() ?? 'generate',
      );
      setState(() {
        _activeJob = {
          'job_id': result['job_id'],
          'status': result['status'],
          'job_type': result['job_type'],
          'progress_message': result['message'],
        };
        _message = result['message']?.toString();
      });
      _syncPolling(_activeJob);
    } catch (e) {
      setState(() => _message = e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _queueing = false);
    }
  }

  Future<void> _modify() async {
    if (_plan == null || _modifyController.text.trim().isEmpty) return;
    final api = ApiService(token: widget.session.token);
    setState(() {
      _queueing = true;
      _message = 'Queueing plan update...';
    });
    try {
      final result = await api.modifyPlan(_plan!['plan_id'] as String, _modifyController.text.trim());
      await widget.session.savePendingPlanJob(
        jobId: result['job_id'] as String,
        jobType: result['job_type']?.toString() ?? 'modify',
      );
      setState(() {
        _activeJob = {
          'job_id': result['job_id'],
          'status': result['status'],
          'job_type': result['job_type'],
          'progress_message': result['message'],
        };
        _message = result['message']?.toString();
        _modifyController.clear();
      });
      _syncPolling(_activeJob);
    } catch (e) {
      setState(() => _message = e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _queueing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) return const Center(child: CircularProgressIndicator());
    final days = (_plan?['days'] as List?)?.cast<Map<String, dynamic>>() ?? [];
    final processing = _activeJob != null && (_activeJob!['status'] == 'queued' || _activeJob!['status'] == 'running');
    return RefreshIndicator(
      onRefresh: _load,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          if (processing)
            SectionCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Row(
                    children: [
                      SizedBox(height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2)),
                      SizedBox(width: 10),
                      Text('Plan pipeline running', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Text(_message ?? 'Generating your personalized plan...'),
                  const SizedBox(height: 8),
                  const Text(
                    'The app stays responsive while the backend pipeline runs. The latest known plan remains visible below until the new result is ready.',
                    style: TextStyle(color: Colors.white70),
                  ),
                ],
              ),
            ),
          SectionCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(_plan?['name']?.toString() ?? 'No current plan', style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                const SizedBox(height: 6),
                Text(
                  _plan?['explanation']?.toString() ?? 'Your plan will appear here after the backend finishes processing.',
                  style: const TextStyle(color: Colors.white70),
                ),
                if (_plan == null && !processing) ...[
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: _queueing ? null : _generate,
                      child: Text(_queueing ? 'Generating...' : 'Generate Plan'),
                    ),
                  ),
                ],
              ],
            ),
          ),
          if (_plan != null)
            ...days.map((day) => SectionCard(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(day['name'].toString(), style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 8),
                      ...((day['exercises'] as List?)?.cast<Map<String, dynamic>>() ?? []).map((ex) => Padding(
                            padding: const EdgeInsets.only(bottom: 12),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(ex['exercise_name'].toString(), style: const TextStyle(fontWeight: FontWeight.w600)),
                                const SizedBox(height: 4),
                                Text(
                                  ((ex['sets'] as List?)?.map((s) => 'set ${s['set_number']}: ${s['target_reps']} reps @ RIR ${s['target_rir']} (${s['rest_seconds']}s)').join('  •  ') ?? ''),
                                  style: const TextStyle(color: Colors.white70),
                                ),
                              ],
                            ),
                          )),
                    ],
                  ),
                )),
          SectionCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Modify this plan', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                const Text(
                  'Updates are queued asynchronously. The current plan stays visible while the modified version is generated.',
                  style: TextStyle(color: Colors.white70),
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: _modifyController,
                  maxLines: 3,
                  decoration: const InputDecoration(
                    labelText: 'Example: Swap lunges because of knee pain, or make this a 3-day plan',
                  ),
                ),
                const SizedBox(height: 12),
                ElevatedButton(onPressed: _plan == null || _queueing ? null : _modify, child: Text(_queueing ? 'Queueing update...' : 'Queue plan update')),
                if (_message != null) ...[
                  const SizedBox(height: 10),
                  Text(_message!, style: const TextStyle(color: Colors.white70)),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}
