import { writeFile } from 'node:fs/promises';
/** Fixed non-interactive question bridge for Pi RPC mode. */
export default function cockpitExtension(pi) {
    if (process.env.B70_RUN_MODE === 'review') {
        pi.registerTool({
            name: 'submit_review',
            label: 'Review abschließen',
            description: 'Speichere den abschließenden strukturierten Reviewbericht. Rufe das Tool genau einmal auf und beende danach den Lauf.',
            parameters: {
                '~kind': 'Object',
                type: 'object',
                properties: {
                    summary: { '~kind': 'String', type: 'string', description: 'Knappe Gesamtbewertung.' },
                    findings: {
                        '~kind': 'Array',
                        type: 'array',
                        maxItems: 100,
                        items: {
                            '~kind': 'Object',
                            type: 'object',
                            properties: {
                                severity: {
                                    '~kind': 'String',
                                    type: 'string',
                                    enum: ['critical', 'high', 'medium', 'low', 'info'],
                                },
                                title: { '~kind': 'String', type: 'string' },
                                file: { '~kind': 'String', type: 'string' },
                                line: { '~kind': 'Number', type: 'integer', minimum: 1 },
                                reasoning: { '~kind': 'String', type: 'string' },
                                evidence: { '~kind': 'String', type: 'string' },
                            },
                            required: ['severity', 'title', 'reasoning'],
                            additionalProperties: false,
                        },
                    },
                },
                required: ['summary', 'findings'],
                additionalProperties: false,
            },
            executionMode: 'sequential',
            async execute(_toolCallId, params) {
                await writeFile('/profile/review.json', JSON.stringify({ version: 1, summary: params.summary, findings: params.findings }), { encoding: 'utf8', flag: 'wx', mode: 0o600 });
                return {
                    content: [
                        {
                            type: 'text',
                            text: 'Der strukturierte Reviewbericht wurde gespeichert. Beende den Lauf jetzt.',
                        },
                    ],
                    details: { findings: params.findings.length, stored: true },
                };
            },
        });
        return;
    }
    pi.registerTool({
        name: 'ask_user',
        label: 'Rückfrage',
        description: 'Stelle genau eine notwendige Rückfrage. Rufe das Tool auf und beende danach sofort den aktuellen Lauf.',
        parameters: {
            '~kind': 'Object',
            type: 'object',
            properties: {
                question: {
                    '~kind': 'String',
                    type: 'string',
                    description: 'Eine konkrete Rückfrage an den Nutzer.',
                },
            },
            required: ['question'],
            additionalProperties: false,
        },
        executionMode: 'sequential',
        async execute(_toolCallId, params) {
            const question = params.question.trim().slice(0, 16_384);
            if (question.length === 0) {
                return { content: [{ type: 'text', text: 'Die Rückfrage darf nicht leer sein.' }] };
            }
            await writeFile('/workspace/.b70-question.json', JSON.stringify({ version: 1, question }), {
                encoding: 'utf8',
                flag: 'wx',
                mode: 0o600,
            });
            return {
                content: [
                    {
                        type: 'text',
                        text: 'Die Rückfrage wurde gespeichert. Beende den Lauf jetzt ohne weitere Toolaufrufe.',
                    },
                ],
                details: { question, waiting: true },
            };
        },
    });
}
