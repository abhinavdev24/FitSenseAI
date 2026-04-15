import 'package:flutter/material.dart';

import '../../services/session_store.dart';
import 'checkin_tab.dart';
import 'coach_tab.dart';
import 'dashboard_tab.dart';
import 'plan_tab.dart';
import 'settings_screen.dart';
import 'workout_tab.dart';

class HomeShell extends StatefulWidget {
  const HomeShell({super.key, required this.session, this.initialIndex = 0});
  final SessionStore session;
  final int initialIndex;

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  late int _index;

  @override
  void initState() {
    super.initState();
    _index = widget.initialIndex;
  }

  @override
  Widget build(BuildContext context) {
    final pages = [
      DashboardTab(session: widget.session),
      PlanTab(session: widget.session),
      WorkoutTab(session: widget.session),
      CheckinTab(session: widget.session),
      CoachTab(session: widget.session),
    ];
    return Scaffold(
      appBar: AppBar(
        title: const Text('FitSense AI'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            onPressed: () {
              Navigator.of(context).push(MaterialPageRoute(builder: (_) => SettingsScreen(session: widget.session)));
            },
          ),
        ],
      ),
      body: pages[_index],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (v) => setState(() => _index = v),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.home_outlined), label: 'Home'),
          NavigationDestination(icon: Icon(Icons.calendar_month_outlined), label: 'Plan'),
          NavigationDestination(icon: Icon(Icons.fitness_center), label: 'Workout'),
          NavigationDestination(icon: Icon(Icons.monitor_weight_outlined), label: 'Check-in'),
          NavigationDestination(icon: Icon(Icons.chat_bubble_outline), label: 'Coach'),
        ],
      ),
    );
  }
}
