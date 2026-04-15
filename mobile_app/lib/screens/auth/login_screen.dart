import 'package:flutter/material.dart';

import '../../app_theme.dart';
import '../../services/api_service.dart';
import '../../services/session_store.dart';
import '../home/home_shell.dart';
import '../onboarding/onboarding_screen.dart';
import 'signup_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key, required this.session});
  final SessionStore session;

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _email = TextEditingController();
  final _password = TextEditingController();
  bool _loading = false;
  String? _error;

  Future<void> _submit() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final api = ApiService(token: null);
      final data = await api.login(_email.text.trim(), _password.text);
      await widget.session.saveAuth(
        token: data['token'] as String,
        email: data['email'] as String,
        name: data['name'] as String,
      );
      if (!mounted) return;
      final needsOnboarding = data['needs_onboarding'] == true;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (_) => needsOnboarding ? OnboardingScreen(session: widget.session) : HomeShell(session: widget.session),
        ),
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
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 420),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.fitness_center, color: AppTheme.primary, size: 56),
                  const SizedBox(height: 16),
                  const Text('FitSense AI', style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  const Text('Sign in to continue your plan, daily check-ins, workouts, and coaching.', style: TextStyle(color: AppTheme.muted)),
                  const SizedBox(height: 28),
                  TextField(controller: _email, decoration: const InputDecoration(labelText: 'Email')),
                  const SizedBox(height: 14),
                  TextField(controller: _password, obscureText: true, decoration: const InputDecoration(labelText: 'Password')),
                  if (_error != null) ...[
                    const SizedBox(height: 12),
                    Text(_error!, style: const TextStyle(color: Colors.redAccent)),
                  ],
                  const SizedBox(height: 20),
                  ElevatedButton(onPressed: _loading ? null : _submit, child: Text(_loading ? 'Signing in...' : 'Log In')),
                  const SizedBox(height: 12),
                  OutlinedButton(
                    onPressed: () {
                      Navigator.of(context).push(MaterialPageRoute(builder: (_) => SignupScreen(session: widget.session)));
                    },
                    child: const Text('Create new account'),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
