import 'package:shared_preferences/shared_preferences.dart';

class SessionStore {
  static const _tokenKey = 'token';
  static const _emailKey = 'email';
  static const _nameKey = 'name';
  static const _pendingPlanJobIdKey = 'pending_plan_job_id';
  static const _pendingPlanJobTypeKey = 'pending_plan_job_type';

  final SharedPreferences prefs;
  SessionStore(this.prefs);

  static Future<SessionStore> create() async {
    final prefs = await SharedPreferences.getInstance();
    return SessionStore(prefs);
  }

  String? get token => prefs.getString(_tokenKey);
  String? get email => prefs.getString(_emailKey);
  String? get name => prefs.getString(_nameKey);
  String? get pendingPlanJobId => prefs.getString(_pendingPlanJobIdKey);
  String? get pendingPlanJobType => prefs.getString(_pendingPlanJobTypeKey);
  bool get isLoggedIn => token != null && token!.isNotEmpty;

  Future<void> saveAuth({required String token, required String email, required String name}) async {
    await prefs.setString(_tokenKey, token);
    await prefs.setString(_emailKey, email);
    await prefs.setString(_nameKey, name);
  }

  Future<void> savePendingPlanJob({required String jobId, required String jobType}) async {
    await prefs.setString(_pendingPlanJobIdKey, jobId);
    await prefs.setString(_pendingPlanJobTypeKey, jobType);
  }

  Future<void> clearPendingPlanJob() async {
    await prefs.remove(_pendingPlanJobIdKey);
    await prefs.remove(_pendingPlanJobTypeKey);
  }

  Future<void> clear() async {
    await prefs.remove(_tokenKey);
    await prefs.remove(_emailKey);
    await prefs.remove(_nameKey);
    await clearPendingPlanJob();
  }
}
