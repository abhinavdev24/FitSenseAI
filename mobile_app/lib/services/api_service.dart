import 'dart:convert';
import 'dart:io' show Platform;

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class ApiService {
  ApiService({required this.token, String? baseUrl}) : baseUrl = baseUrl ?? defaultBaseUrl;

  final String? token;
  final String baseUrl;

  static String get defaultBaseUrl {
    // Pass --dart-define=BACKEND_URL=https://your-backend.com for production builds.
    const envUrl = String.fromEnvironment('BACKEND_URL', defaultValue: '');
    if (envUrl.isNotEmpty) return envUrl;
    if (kIsWeb) return 'http://127.0.0.1:8000';
    try {
      if (Platform.isAndroid) return 'http://10.0.2.2:8000';
      return 'http://127.0.0.1:8000';
    } catch (_) {
      return 'http://127.0.0.1:8000';
    }
  }

  Map<String, String> get _headers => {
        'Content-Type': 'application/json',
        if (token != null) 'Authorization': 'Bearer $token',
      };

  Future<Map<String, dynamic>> _request(String method, String path, {Map<String, dynamic>? body}) async {
    final uri = Uri.parse('$baseUrl$path');
    late http.Response response;
    final encoded = body == null ? null : jsonEncode(body);
    switch (method) {
      case 'GET':
        response = await http.get(uri, headers: _headers);
        break;
      case 'POST':
        response = await http.post(uri, headers: _headers, body: encoded);
        break;
      default:
        throw Exception('Unsupported method: $method');
    }
    if (response.statusCode >= 400) {
      throw Exception(_errorMessage(response));
    }
    if (response.body.isEmpty) return {};
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  String _errorMessage(http.Response response) {
    try {
      final data = jsonDecode(response.body);
      if (data is Map && data['detail'] != null) return data['detail'].toString();
    } catch (_) {}
    return 'Request failed (${response.statusCode})';
  }

  Future<Map<String, dynamic>> signup(String name, String email, String password) =>
      _request('POST', '/auth/signup', body: {'name': name, 'email': email, 'password': password});

  Future<Map<String, dynamic>> login(String email, String password) =>
      _request('POST', '/auth/login', body: {'email': email, 'password': password});

  Future<Map<String, dynamic>> me() => _request('GET', '/me');
  Future<Map<String, dynamic>> dashboard() => _request('GET', '/dashboard');
  Future<Map<String, dynamic>> currentPlan() => _request('GET', '/plans/current');
  Future<Map<String, dynamic>> recentWorkouts() => _request('GET', '/workouts/recent');
  Future<Map<String, dynamic>> latestPlanJob() => _request('GET', '/plans/jobs/latest');
  Future<Map<String, dynamic>> planJobStatus(String jobId) => _request('GET', '/plans/jobs/$jobId');

  Future<Map<String, dynamic>> saveOnboarding(Map<String, dynamic> payload) =>
      _request('POST', '/profile/onboarding', body: payload);

  Future<Map<String, dynamic>> createPlan(Map<String, dynamic> payload) =>
      _request('POST', '/plans', body: payload);

  Future<Map<String, dynamic>> triggerPipeline(Map<String, dynamic> payload) =>
      _request('POST', '/pipeline/trigger', body: payload);

  Future<Map<String, dynamic>> modifyPlan(String planId, String instruction) =>
      _request('POST', '/plans/$planId:modify', body: {'instruction': instruction});

  Future<Map<String, dynamic>> createWorkout(Map<String, dynamic> payload) =>
      _request('POST', '/workouts', body: payload);

  Future<Map<String, dynamic>> createWorkoutExercise(String workoutId, Map<String, dynamic> payload) =>
      _request('POST', '/workouts/$workoutId/exercises', body: payload);

  Future<Map<String, dynamic>> createWorkoutSet(String workoutId, Map<String, dynamic> payload) =>
      _request('POST', '/workouts/$workoutId/sets', body: payload);

  Future<Map<String, dynamic>> logSleep(DateTime date, double hours) => _request('POST', '/daily/sleep', body: {
        'logged_on': date.toIso8601String().split('T').first,
        'hours': hours,
      });

  Future<Map<String, dynamic>> logCalories(DateTime date, int calories) => _request('POST', '/daily/calories', body: {
        'logged_on': date.toIso8601String().split('T').first,
        'calories': calories,
      });

  Future<Map<String, dynamic>> logWeight(DateTime when, double weightKg) => _request('POST', '/daily/weight', body: {
        'logged_at': when.toIso8601String(),
        'weight_kg': weightKg,
      });

  Future<Map<String, dynamic>> coach(String message) =>
      _request('POST', '/coach', body: {'message': message, 'context_mode': 'short'});

  Future<Map<String, dynamic>> adaptation() =>
      _request('POST', '/adaptation:next_week', body: {'days_window': 14});

  Future<Map<String, dynamic>> setCloudEndpoint(String url) =>
      _request('POST', '/cloud/set-endpoint', body: {'url': url});

  Future<Map<String, dynamic>> cloudStatus() =>
      _request('GET', '/cloud/status');
}
