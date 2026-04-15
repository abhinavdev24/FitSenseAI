import 'package:flutter/material.dart';

import '../../services/api_service.dart';
import '../../services/session_store.dart';

class CoachTab extends StatefulWidget {
  const CoachTab({super.key, required this.session});
  final SessionStore session;

  @override
  State<CoachTab> createState() => _CoachTabState();
}

class _CoachTabState extends State<CoachTab> {
  final _controller = TextEditingController();
  final List<Map<String, String>> _messages = [
    {
      'role': 'assistant',
      'text': 'Hey! I\'m your AI fitness coach. Ask me anything — exercise swaps, how to progress your lifts, recovery tips, or how to adjust your plan around injuries or a busy week.'
    }
  ];
  bool _sending = false;

  Future<void> _send() async {
    final text = _controller.text.trim();
    if (text.isEmpty) return;
    setState(() {
      _messages.add({'role': 'user', 'text': text});
      _controller.clear();
      _sending = true;
    });
    try {
      final api = ApiService(token: widget.session.token);
      final res = await api.coach(text);
      setState(() => _messages.add({'role': 'assistant', 'text': res['reply'].toString()}));
    } catch (e) {
      setState(() => _messages.add({'role': 'assistant', 'text': e.toString().replaceFirst('Exception: ', '')}));
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.all(16),
            itemCount: _messages.length,
            itemBuilder: (context, index) {
              final msg = _messages[index];
              final isUser = msg['role'] == 'user';
              return Align(
                alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                child: Container(
                  constraints: const BoxConstraints(maxWidth: 520),
                  margin: const EdgeInsets.only(bottom: 10),
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: isUser ? Theme.of(context).colorScheme.primary : Theme.of(context).cardColor,
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Text(msg['text'] ?? ''),
                ),
              );
            },
          ),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
          child: Row(
            children: [
              Expanded(child: TextField(controller: _controller, minLines: 1, maxLines: 4, decoration: const InputDecoration(labelText: 'Ask your coach'))),
              const SizedBox(width: 8),
              FilledButton(onPressed: _sending ? null : _send, child: Text(_sending ? '...' : 'Send')),
            ],
          ),
        ),
      ],
    );
  }
}
