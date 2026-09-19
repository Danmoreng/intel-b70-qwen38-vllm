import { spawnSync } from 'node:child_process';
import { readFileSync, rmSync } from 'node:fs';
import { resolve } from 'node:path';
const MAX_OUTPUT = 512 * 1024;
const MAX_CHECK_OUTPUT = 64 * 1024;
const safeRevision = /^[a-f0-9]{40}$/;
const safeRelative = /^(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$))[^\0]+$/;
function fail(message) {
    process.stdout.write(JSON.stringify({ ok: false, error: message.slice(0, 512) }));
    process.exit(1);
}
function git(args, cwd = '/workspace', timeout = 120_000) {
    return spawnSync('/usr/bin/git', [
        '-c',
        'core.hooksPath=/dev/null',
        '-c',
        'commit.gpgSign=false',
        '-c',
        'core.pager=cat',
        ...args,
    ], { cwd, encoding: 'utf8', timeout, maxBuffer: MAX_OUTPUT });
}
function cleanText(value) {
    return String(value ?? '').slice(0, MAX_OUTPUT);
}
function decode() {
    const encoded = process.argv[2];
    if (encoded === undefined || encoded.length > 16_384)
        fail('Ungültige Utility-Konfiguration.');
    try {
        const value = JSON.parse(Buffer.from(encoded, 'base64url').toString('utf8'));
        if (!safeRevision.test(value.baseRevision))
            fail('Ungültige Basisrevision.');
        return value;
    }
    catch {
        fail('Ungültige Utility-Konfiguration.');
    }
}
function changedPaths(baseRevision) {
    const tracked = git([
        'diff',
        '--name-only',
        '--no-ext-diff',
        '--no-textconv',
        `${baseRevision}..HEAD`,
    ]);
    if (tracked.status !== 0)
        fail('Geänderte Dateien konnten nicht ermittelt werden.');
    return cleanText(tracked.stdout)
        .split('\n')
        .map((value) => value.trim())
        .filter(Boolean);
}
function pathAllowed(path, roots) {
    return roots.some((root) => root === '.' || path === root || path.startsWith(`${root}/`));
}
function finalize(config) {
    const questionPath = '/workspace/.b70-question.json';
    let question;
    try {
        const value = JSON.parse(readFileSync(questionPath, 'utf8'));
        if (typeof value.question === 'string' && value.question.trim())
            question = value.question.trim().slice(0, 16_384);
    }
    catch {
        // The normal completion path has no question marker.
    }
    rmSync(questionPath, { force: true });
    const status = git(['status', '--porcelain=v1', '-z']);
    if (status.status !== 0)
        fail('Git-Status konnte nicht gelesen werden.');
    if (cleanText(status.stdout).length > 0) {
        const add = git(['add', '-A']);
        if (add.status !== 0)
            fail(`Arbeitsstand konnte nicht gesichert werden: ${cleanText(add.stderr)}`);
        const commit = git([
            '-c',
            'user.name=Local AI B70',
            '-c',
            'user.email=agent@local.invalid',
            'commit',
            '-m',
            'checkpoint: local agent result',
        ]);
        if (commit.status !== 0)
            fail(`Checkpoint konnte nicht erstellt werden: ${cleanText(commit.stderr)}`);
    }
    const revisionResult = git(['rev-parse', 'HEAD']);
    const resultRevision = cleanText(revisionResult.stdout).trim();
    if (revisionResult.status !== 0 || !safeRevision.test(resultRevision))
        fail('Ergebnisrevision konnte nicht bestimmt werden.');
    const files = changedPaths(config.baseRevision);
    const editPaths = config.editPaths?.length ? config.editPaths : ['.'];
    if (editPaths.some((path) => !safeRelative.test(path)))
        fail('Ungültiger Bearbeitungsbereich.');
    const forbidden = files.filter((path) => !pathAllowed(path, editPaths));
    if (forbidden.length > 0)
        fail(`Bearbeitungsbereich überschritten: ${forbidden.slice(0, 10).join(', ')}`);
    const diff = git([
        'diff',
        '--binary',
        '--no-ext-diff',
        '--no-textconv',
        '--stat',
        '--patch',
        `${config.baseRevision}..${resultRevision}`,
    ]);
    if (diff.status !== 0)
        fail('Diff konnte nicht erzeugt werden.');
    const cwd = resolve('/workspace', config.workdir ?? '.');
    if (cwd !== '/workspace' && !cwd.startsWith('/workspace/'))
        fail('Ungültiges Arbeitsverzeichnis.');
    const checks = (config.checks ?? []).slice(0, 12).map((check) => {
        const started = Date.now();
        if (!check.label || !Array.isArray(check.argv) || check.argv.length === 0) {
            return {
                label: check.label || 'Check',
                argv: check.argv ?? [],
                status: 'failed',
                exitCode: null,
                durationMs: 0,
                output: 'Ungültiger Check.',
            };
        }
        const result = spawnSync(check.argv[0], check.argv.slice(1), {
            cwd,
            encoding: 'utf8',
            timeout: 10 * 60_000,
            maxBuffer: MAX_CHECK_OUTPUT * 2,
            env: { HOME: '/tmp', PATH: '/usr/bin:/bin', LANG: 'C.UTF-8', CI: '1' },
        });
        return {
            label: check.label.slice(0, 160),
            argv: check.argv,
            status: result.status === 0 ? 'passed' : 'failed',
            exitCode: result.status,
            durationMs: Date.now() - started,
            output: `${cleanText(result.stdout)}${cleanText(result.stderr)}`.slice(0, MAX_CHECK_OUTPUT),
        };
    });
    process.stdout.write(JSON.stringify({
        ok: true,
        resultRevision,
        files,
        diff: cleanText(diff.stdout),
        checks,
        checkStatus: checks.length === 0
            ? 'not_run'
            : checks.every((check) => check.status === 'passed')
                ? 'passed'
                : 'failed',
        ...(question === undefined ? {} : { question }),
    }));
}
function accept(config) {
    if (!safeRevision.test(config.resultRevision ?? '') || !config.sourceWorkspace)
        fail('Ergebnisrevision oder Quellworkspace fehlt.');
    const current = cleanText(git(['rev-parse', 'HEAD']).stdout).trim();
    if (current !== config.baseRevision)
        fail('Der Zielbranch hat sich seit Aufgabenstart verändert.');
    if (cleanText(git(['status', '--porcelain=v1']).stdout).trim())
        fail('Die Zielarbeitskopie enthält nicht übernommene Änderungen.');
    const filters = git([
        'config',
        '--local',
        '--get-regexp',
        '^filter[.].*[.](clean|smudge|process)$',
    ]);
    if (filters.status === 0 && cleanText(filters.stdout).trim()) {
        fail('Die Zielarbeitskopie verwendet ausführbare Git-Filter; automatische Übernahme ist gesperrt.');
    }
    const fetch = git(['fetch', '--no-tags', config.sourceWorkspace, config.resultRevision]);
    if (fetch.status !== 0)
        fail(`Ergebnis konnte nicht importiert werden: ${cleanText(fetch.stderr)}`);
    const ancestry = git(['merge-base', '--is-ancestor', config.baseRevision, 'FETCH_HEAD']);
    if (ancestry.status !== 0)
        fail('Das Ergebnis basiert nicht auf der gespeicherten Zielbasis.');
    const merge = git(['merge', '--ff-only', '--no-stat', 'FETCH_HEAD']);
    if (merge.status !== 0)
        fail(`Ergebnis konnte nicht übernommen werden: ${cleanText(merge.stderr)}`);
    process.stdout.write(JSON.stringify({ ok: true, acceptedRevision: config.resultRevision }));
}
const config = decode();
if (config.mode === 'finalize')
    finalize(config);
else if (config.mode === 'accept')
    accept(config);
else
    fail('Unbekannter Utility-Modus.');
