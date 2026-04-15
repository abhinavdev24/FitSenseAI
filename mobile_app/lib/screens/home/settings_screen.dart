import 'package:flutter/material.dart';

import '../../services/api_service.dart';
import '../../services/session_store.dart';
import '../auth/login_screen.dart';
import '../../widgets/common.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key, required this.session});
  final SessionStore session;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  Map<String, dynamic>? _profile;
  Map<String, dynamic>? _adaptation;
  Map<String, dynamic>? _cloudStatus;
  String? _message;
  bool _loading = true;
  final _ngrokController = TextEditingController();
  bool _cloudSaving = false;
  String? _cloudMessage;

  @override
  void initState() {
    super.initState();
    _load();
    _loadCloudStatus();
  }

  @override
  void dispose() {
    _ngrokController.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final api = ApiService(token: widget.session.token);
      final profile = await api.me();
      if (!mounted) return;
      setState(() => _profile = profile);
    } catch (e) {
      setState(() => _message = e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _loadCloudStatus() async {
    try {
      final api = ApiService(token: widget.session.token);
      final status = await api.cloudStatus();
      if (!mounted) return;
      setState(() {
        _cloudStatus = status;
        final url = status['cloud_predict_url'] as String? ?? '';
        // Strip /predict suffix to show the base ngrok URL
        _ngrokController.text = url.endsWith('/predict') ? url.substring(0, url.length - 8) : url;
      });
    } catch (_) {}
  }

  Future<void> _saveCloudEndpoint() async {
    final url = _ngrokController.text.trim();
    if (url.isEmpty) return;
    setState(() {
      _cloudSaving = true;
      _cloudMessage = null;
    });
    try {
      final api = ApiService(token: widget.session.token);
      final result = await api.setCloudEndpoint(url);
      if (!mounted) return;
      setState(() {
        _cloudMessage = result['message']?.toString() ?? 'Updated';
        _cloudStatus = result;
      });
    } catch (e) {
      setState(() => _cloudMessage = e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _cloudSaving = false);
    }
  }

  Future<void> _runAdaptation() async {
    try {
      final api = ApiService(token: widget.session.token);
      final result = await api.adaptation();
      setState(() => _adaptation = result);
    } catch (e) {
      setState(() => _message = e.toString().replaceFirst('Exception: ', ''));
    }
  }

  Future<void> _logout() async {
    await widget.session.clear();
    if (!mounted) return;
    Navigator.of(context).pushAndRemoveUntil(
      MaterialPageRoute(builder: (_) => LoginScreen(session: widget.session)),
      (route) => false,
    );
  }

  Widget _profileRow(String label, String? value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 130,
            child: Text(label, style: const TextStyle(color: Colors.white54, fontSize: 13)),
          ),
          Expanded(
            child: Text(
              value ?? '—',
              style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings & history')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : ListView(
              padding: const EdgeInsets.all(16),
              children: [
                SectionCard(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(widget.session.name ?? 'User', style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 4),
                      Text(widget.session.email ?? '', style: const TextStyle(color: Colors.white70)),
                    ],
                  ),
                ),
                if (_profile != null)
                  SectionCard(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('Your profile', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                        const SizedBox(height: 12),
                        _profileRow('Goal', _profile!['goal']?.toString()),
                        _profileRow('Training days', _profile!['days_per_week']?.toString() != null ? '${_profile!['days_per_week']} days / week' : null),
                        _profileRow('Experience', _profile!['experience_level']?.toString()),
                        _profileRow('Equipment', (_profile!['equipment'] as List?)?.join(', ')),
                        _profileRow('Injuries / notes', (_profile!['injuries'] as String?)?.isNotEmpty == true ? _profile!['injuries'].toString() : 'None'),
                        _profileRow('Conditions', (_profile!['conditions'] as List?)?.isNotEmpty == true ? (_profile!['conditions'] as List).join(', ') : 'None'),
                      ],
                    ),
                  ),
                SectionCard(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('Cloud inference (Colab / ngrok)', style: TextStyle(fontWeight: FontWeight.bold)),
                      const SizedBox(height: 6),
                      if (_cloudStatus != null) ...[
                        Row(
                          children: [
                            Icon(
                              (_cloudStatus!['can_use_cloud'] == true) ? Icons.cloud_done : Icons.cloud_off,
                              size: 16,
                              color: (_cloudStatus!['can_use_cloud'] == true) ? Colors.greenAccent : Colors.white38,
                            ),
                            const SizedBox(width: 6),
                            Text(
                              (_cloudStatus!['can_use_cloud'] == true) ? 'Cloud online' : 'Cloud offline / not set',
                              style: TextStyle(
                                color: (_cloudStatus!['can_use_cloud'] == true) ? Colors.greenAccent : Colors.white38,
                                fontSize: 13,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 4),
                      ],
                      const Text(
                        'Paste your ngrok URL here (e.g. https://xxxx.ngrok-free.app). The backend will start routing plan generation and coach through your Colab model.',
                        style: TextStyle(color: Colors.white54, fontSize: 12),
                      ),
                      const SizedBox(height: 10),
                      TextField(
                        controller: _ngrokController,
                        decoration: const InputDecoration(
                          labelText: 'ngrok URL',
                          hintText: 'https://xxxx.ngrok-free.app',
                        ),
                      ),
                      const SizedBox(height: 10),
                      ElevatedButton(
                        onPressed: _cloudSaving ? null : _saveCloudEndpoint,
                        child: Text(_cloudSaving ? 'Updating...' : 'Set endpoint'),
                      ),
                      if (_cloudMessage != null) ...[
                        const SizedBox(height: 8),
                        Text(_cloudMessage!, style: const TextStyle(color: Colors.greenAccent, fontSize: 13)),
                      ],
                    ],
                  ),
                ),
                SectionCard(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('Next-week adaptation', style: TextStyle(fontWeight: FontWeight.bold)),
                      const SizedBox(height: 8),
                      OutlinedButton(onPressed: _runAdaptation, child: const Text('Compute adaptation summary')),
                      if (_adaptation != null) ...[
                        const SizedBox(height: 10),
                        Text(_adaptation.toString()),
                      ],
                    ],
                  ),
                ),
                if (_message != null) Text(_message!, style: const TextStyle(color: Colors.redAccent)),
                const SizedBox(height: 10),
                ElevatedButton(onPressed: _logout, child: const Text('Log out')),
              ],
            ),
    );
  }
}
