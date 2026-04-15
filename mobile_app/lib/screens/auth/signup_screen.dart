import 'package:flutter/material.dart';

import '../../services/api_service.dart';
import '../../services/session_store.dart';
import '../onboarding/onboarding_screen.dart';

class SignupScreen extends StatefulWidget {
  const SignupScreen({super.key, required this.session});
  final SessionStore session;

  @override
  State<SignupScreen> createState() => _SignupScreenState();
}

class _SignupScreenState extends State<SignupScreen> {
  final _name = TextEditingController();
  final _email = TextEditingController();
  final _password = TextEditingController();
  bool _loading = false;
  bool _emailAlreadyExists = false;
  String? _error;

  Future<void> _submit() async {
    setState(() {
      _loading = true;
      _error = null;
      _emailAlreadyExists = false;
    });
    try {
      final api = ApiService(token: null);
      final data = await api.signup(_name.text.trim(), _email.text.trim(), _password.text);
      await widget.session.saveAuth(
        token: data['token'] as String,
        email: data['email'] as String,
        name: data['name'] as String,
      );
      if (!mounted) return;
      Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (_) => OnboardingScreen(session: widget.session)));
    } catch (e) {
      final msg = e.toString().replaceFirst('Exception: ', '');
      setState(() {
        _emailAlreadyExists = msg.toLowerCase().contains('already registered');
        _error = _emailAlreadyExists ? null : msg;
      });
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 420),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Create account', style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                const Text('Enter your details to get started.', style: TextStyle(color: Colors.white70)),
                const SizedBox(height: 24),
                TextField(controller: _name, decoration: const InputDecoration(labelText: 'Full name')),
                const SizedBox(height: 14),
                TextField(
                  controller: _email,
                  keyboardType: TextInputType.emailAddress,
                  autocorrect: false,
                  decoration: const InputDecoration(labelText: 'Email'),
                ),
                const SizedBox(height: 14),
                TextField(controller: _password, obscureText: true, decoration: const InputDecoration(labelText: 'Password (min 6 chars)')),
                if (_emailAlreadyExists) ...[
                  const SizedBox(height: 14),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.orange.withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.orange.withValues(alpha: 0.4)),
                    ),
                    child: Row(
                      children: [
                        const Icon(Icons.info_outline, color: Colors.orange, size: 18),
                        const SizedBox(width: 8),
                        const Expanded(child: Text('This email is already registered.', style: TextStyle(color: Colors.orange))),
                        TextButton(
                          onPressed: () => Navigator.of(context).pop(),
                          style: TextButton.styleFrom(foregroundColor: Colors.orange, padding: const EdgeInsets.symmetric(horizontal: 8)),
                          child: const Text('Log in instead'),
                        ),
                      ],
                    ),
                  ),
                ],
                if (_error != null) ...[
                  const SizedBox(height: 12),
                  Text(_error!, style: const TextStyle(color: Colors.redAccent)),
                ],
                const SizedBox(height: 20),
                ElevatedButton(onPressed: _loading ? null : _submit, child: Text(_loading ? 'Creating account...' : 'Continue')),
                const SizedBox(height: 12),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text('Already have an account?', style: TextStyle(color: Colors.white70)),
                    TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text('Log in')),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
