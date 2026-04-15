import 'package:flutter/material.dart';

import 'app_theme.dart';
import 'services/session_store.dart';
import 'screens/auth/login_screen.dart';
import 'screens/home/home_shell.dart';

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
    return MaterialApp(
      title: 'FitSense AI',
      theme: AppTheme.darkTheme(),
      debugShowCheckedModeBanner: false,
      home: session.isLoggedIn
          ? HomeShell(session: session)
          : LoginScreen(session: session),
    );
  }
}
