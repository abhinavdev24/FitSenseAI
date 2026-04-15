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
  String? _message;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
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
                      const SizedBox(height: 6),
                      Text(widget.session.email ?? ''),
                      const SizedBox(height: 10),
                      Text('Backend URL: ${ApiService.defaultBaseUrl}', style: const TextStyle(color: Colors.white70)),
                    ],
                  ),
                ),
                if (_profile != null)
                  SectionCard(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('Stored profile', style: TextStyle(fontWeight: FontWeight.bold)),
                        const SizedBox(height: 8),
                        Text(_profile.toString()),
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
