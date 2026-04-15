import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:fitsense_ai/main.dart';
import 'package:fitsense_ai/services/session_store.dart';

void main() {
  testWidgets('app renders login when no session exists', (tester) async {
    SharedPreferences.setMockInitialValues({});
    final session = await SessionStore.create();
    await tester.pumpWidget(FitSenseApp(session: session));
    expect(find.text('FitSense AI'), findsOneWidget);
  });
}
