import 'package:flutter/material.dart';

import '../../app_theme.dart';
import '../../services/api_service.dart';
import '../../services/session_store.dart';
import '../auth/login_screen.dart';

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

  void _logout() async {
    await widget.session.clear();
    if (!mounted) return;
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (_) => LoginScreen(session: widget.session)),
    );
  }

  @override
  Widget build(BuildContext context) {
    final api = ApiService(token: widget.session.token);
    final tabs = [
      _DashboardTab(api: api, session: widget.session),
      _PlanTab(api: api, session: widget.session),
      _CoachTab(api: api),
      _ProfileTab(session: widget.session, onLogout: _logout),
    ];

    return Scaffold(
      body: tabs[_index],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _index,
        onTap: (i) => setState(() => _index = i),
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.dashboard_outlined), activeIcon: Icon(Icons.dashboard), label: 'Home'),
          BottomNavigationBarItem(icon: Icon(Icons.fitness_center_outlined), activeIcon: Icon(Icons.fitness_center), label: 'Plan'),
          BottomNavigationBarItem(icon: Icon(Icons.chat_bubble_outline), activeIcon: Icon(Icons.chat_bubble), label: 'Coach'),
          BottomNavigationBarItem(icon: Icon(Icons.person_outline), activeIcon: Icon(Icons.person), label: 'Profile'),
        ],
      ),
    );
  }
}

// ── Dashboard ────────────────────────────────────────────────────────────────
class _DashboardTab extends StatefulWidget {
  const _DashboardTab({required this.api, required this.session});
  final ApiService api;
  final SessionStore session;

  @override
  State<_DashboardTab> createState() => _DashboardTabState();
}

class _DashboardTabState extends State<_DashboardTab> {
  Map<String, dynamic>? _data;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final d = await widget.api.dashboard();
      if (mounted) setState(() => _data = d);
    } catch (e) {
      if (mounted) setState(() => _error = e.toString().replaceFirst('Exception: ', ''));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Hi, ${widget.session.name ?? 'there'} 👋')),
      body: RefreshIndicator(
        onRefresh: _load,
        child: _error != null
            ? Center(child: Text(_error!, style: const TextStyle(color: Colors.redAccent)))
            : _data == null
                ? const Center(child: CircularProgressIndicator(color: AppTheme.primary))
                : ListView(
                    padding: const EdgeInsets.all(16),
                    children: [
                      _SummaryCard(data: _data!),
                      const SizedBox(height: 16),
                      _DailyLogCard(api: widget.api),
                    ],
                  ),
      ),
    );
  }
}

class _SummaryCard extends StatelessWidget {
  const _SummaryCard({required this.data});
  final Map<String, dynamic> data;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('This Week', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _Stat('Workouts', '${data['workouts_this_week'] ?? 0}'),
                _Stat('Adherence', '${data['adherence_pct'] ?? 0}%'),
                _Stat('Streak', '${data['streak_days'] ?? 0} days'),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat(this.label, this.value);
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(value, style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: AppTheme.primary)),
        Text(label, style: const TextStyle(color: AppTheme.muted, fontSize: 12)),
      ],
    );
  }
}

class _DailyLogCard extends StatefulWidget {
  const _DailyLogCard({required this.api});
  final ApiService api;

  @override
  State<_DailyLogCard> createState() => _DailyLogCardState();
}

class _DailyLogCardState extends State<_DailyLogCard> {
  final _sleepCtrl = TextEditingController();
  final _calCtrl = TextEditingController();
  final _weightCtrl = TextEditingController();
  bool _saving = false;

  Future<void> _save() async {
    setState(() => _saving = true);
    try {
      final now = DateTime.now();
      if (_sleepCtrl.text.isNotEmpty) await widget.api.logSleep(now, double.parse(_sleepCtrl.text));
      if (_calCtrl.text.isNotEmpty) await widget.api.logCalories(now, int.parse(_calCtrl.text));
      if (_weightCtrl.text.isNotEmpty) await widget.api.logWeight(now, double.parse(_weightCtrl.text));
      _sleepCtrl.clear(); _calCtrl.clear(); _weightCtrl.clear();
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Daily log saved ✓')));
    } catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.toString())));
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Daily Log', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            const SizedBox(height: 12),
            TextField(controller: _sleepCtrl, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Sleep (hours)')),
            const SizedBox(height: 10),
            TextField(controller: _calCtrl, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Calories')),
            const SizedBox(height: 10),
            TextField(controller: _weightCtrl, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Body weight (kg)')),
            const SizedBox(height: 14),
            ElevatedButton(onPressed: _saving ? null : _save, child: Text(_saving ? 'Saving...' : 'Save Today')),
          ],
        ),
      ),
    );
  }
}

// ── Plan ──────────────────────────────────────────────────────────────────────
class _PlanTab extends StatefulWidget {
  const _PlanTab({required this.api, required this.session});
  final ApiService api;
  final SessionStore session;

  @override
  State<_PlanTab> createState() => _PlanTabState();
}

class _PlanTabState extends State<_PlanTab> {
  Map<String, dynamic>? _plan;
  String? _error;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() { _loading = true; _error = null; });
    try {
      final p = await widget.api.currentPlan();
      if (mounted) setState(() { _plan = p; _loading = false; });
    } catch (e) {
      if (mounted) setState(() { _error = e.toString().replaceFirst('Exception: ', ''); _loading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('My Plan')),
      body: RefreshIndicator(
        onRefresh: _load,
        child: _loading
            ? const Center(child: CircularProgressIndicator(color: AppTheme.primary))
            : _error != null
                ? Center(child: Text(_error!, style: const TextStyle(color: Colors.redAccent)))
                : _plan == null || _plan!.isEmpty
                    ? const Center(child: Text('No plan yet. Complete onboarding to generate one.', style: TextStyle(color: AppTheme.muted)))
                    : ListView(
                        padding: const EdgeInsets.all(16),
                        children: [
                          Card(
                            child: Padding(
                              padding: const EdgeInsets.all(16),
                              child: Text(
                                _plan!['summary'] as String? ?? _plan.toString(),
                                style: const TextStyle(fontSize: 14, height: 1.6),
                              ),
                            ),
                          ),
                        ],
                      ),
      ),
    );
  }
}

// ── Coach ─────────────────────────────────────────────────────────────────────
class _CoachTab extends StatefulWidget {
  const _CoachTab({required this.api});
  final ApiService api;

  @override
  State<_CoachTab> createState() => _CoachTabState();
}

class _CoachTabState extends State<_CoachTab> {
  final _ctrl = TextEditingController();
  final _scroll = ScrollController();
  final List<_Msg> _msgs = [];
  bool _loading = false;

  Future<void> _send() async {
    final text = _ctrl.text.trim();
    if (text.isEmpty) return;
    _ctrl.clear();
    setState(() { _msgs.add(_Msg(text, true)); _loading = true; });
    try {
      final res = await widget.api.coach(text);
      final reply = res['response'] as String? ?? res['message'] as String? ?? res.toString();
      if (mounted) setState(() => _msgs.add(_Msg(reply, false)));
    } catch (e) {
      if (mounted) setState(() => _msgs.add(_Msg(e.toString().replaceFirst('Exception: ', ''), false)));
    } finally {
      if (mounted) setState(() => _loading = false);
      await Future.delayed(const Duration(milliseconds: 100));
      _scroll.animateTo(_scroll.position.maxScrollExtent, duration: const Duration(milliseconds: 300), curve: Curves.easeOut);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('AI Coach')),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              controller: _scroll,
              padding: const EdgeInsets.all(16),
              itemCount: _msgs.length,
              itemBuilder: (_, i) {
                final m = _msgs[i];
                return Align(
                  alignment: m.isUser ? Alignment.centerRight : Alignment.centerLeft,
                  child: Container(
                    margin: const EdgeInsets.only(bottom: 8),
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                    constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.78),
                    decoration: BoxDecoration(
                      color: m.isUser ? AppTheme.primary : AppTheme.surface,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(m.text, style: TextStyle(color: m.isUser ? Colors.black : Colors.white, fontSize: 14)),
                  ),
                );
              },
            ),
          ),
          if (_loading) const LinearProgressIndicator(color: AppTheme.primary),
          Container(
            padding: const EdgeInsets.fromLTRB(12, 8, 12, 16),
            color: AppTheme.surface,
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _ctrl,
                    decoration: const InputDecoration(hintText: 'Ask your coach...', border: InputBorder.none),
                    onSubmitted: (_) => _send(),
                  ),
                ),
                IconButton(onPressed: _loading ? null : _send, icon: const Icon(Icons.send, color: AppTheme.primary)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _Msg {
  _Msg(this.text, this.isUser);
  final String text;
  final bool isUser;
}

// ── Profile ───────────────────────────────────────────────────────────────────
class _ProfileTab extends StatelessWidget {
  const _ProfileTab({required this.session, required this.onLogout});
  final SessionStore session;
  final VoidCallback onLogout;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Profile')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(session.name ?? '', style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 4),
                  Text(session.email ?? '', style: const TextStyle(color: AppTheme.muted)),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          OutlinedButton(onPressed: onLogout, child: const Text('Log Out')),
        ],
      ),
    );
  }
}
