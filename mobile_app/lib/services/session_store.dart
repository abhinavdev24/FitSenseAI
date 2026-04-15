import 'package:shared_preferences/shared_preferences.dart';

class SessionStore {
  SessionStore._(this._prefs);

  final SharedPreferences _prefs;

  static Future<SessionStore> create() async {
    final prefs = await SharedPreferences.getInstance();
    return SessionStore._(prefs);
  }

  bool get isLoggedIn => _prefs.getString('token') != null;
  String? get token => _prefs.getString('token');
  String? get email => _prefs.getString('email');
  String? get name => _prefs.getString('name');
  String? get pendingPlanJobId => _prefs.getString('pending_plan_job_id');

  Future<void> saveAuth({required String token, required String email, required String name}) async {
    await _prefs.setString('token', token);
    await _prefs.setString('email', email);
    await _prefs.setString('name', name);
  }

  Future<void> setPendingPlanJobId(String jobId) async {
    await _prefs.setString('pending_plan_job_id', jobId);
  }

  Future<void> clearPendingPlanJobId() async {
    await _prefs.remove('pending_plan_job_id');
  }

  Future<void> clear() async {
    await _prefs.clear();
  }
}
