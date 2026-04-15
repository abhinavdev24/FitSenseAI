import 'package:flutter/material.dart';

import 'app_theme.dart';
import 'screens/auth/login_screen.dart';
import 'screens/home/home_shell.dart';
import 'services/session_store.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final session = await SessionStore.create();
  runApp(FitSenseApp(session: session));
}

class FitSenseApp extends StatelessWidget {
  const FitSenseApp({super.key, required this.session});
  final SessionStore session;

  @override
  Widget build(BuildContext context) {
    final initialIndex = session.pendingPlanJobId != null ? 1 : 0;
    return MaterialApp(
      title: 'FitSense AI',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.darkTheme(),
      home: session.isLoggedIn ? HomeShell(session: session, initialIndex: initialIndex) : LoginScreen(session: session),
    );
  }
}
