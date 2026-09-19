import { execFile } from 'node:child_process';
import { mkdirSync, realpathSync, rmSync, statSync } from 'node:fs';
import { isAbsolute, join, relative, sep } from 'node:path';
const GIT = '/usr/bin/git';
const COMMIT_HASH = /^[a-f0-9]{40}$/;
const RUN_ID = /^[a-f0-9]{8,64}$/;
function canonicalDirectory(path, label) {
    if (!isAbsolute(path) || path.includes('\0'))
        throw new Error(`${label} must be absolute`);
    const canonical = realpathSync(path);
    if (!statSync(canonical).isDirectory())
        throw new Error(`${label} must be a directory`);
    return canonical;
}
function isWithin(parent, child) {
    const suffix = relative(parent, child);
    return (suffix === '' || (!suffix.startsWith(`..${sep}`) && suffix !== '..' && !isAbsolute(suffix)));
}
async function resolveCommit(repository, revision) {
    if (!COMMIT_HASH.test(revision))
        throw new Error('revision must be a full lowercase commit hash');
    return await new Promise((resolvePromise, reject) => {
        execFile(GIT, ['-C', repository, 'rev-parse', '--verify', `${revision}^{commit}`], { timeout: 10_000, maxBuffer: 1024 }, (error, stdout) => {
            if (error !== null)
                reject(new Error('revision is not a commit in the source repository'));
            else
                resolvePromise(stdout.trim());
        });
    });
}
function git(args, timeout = 60_000) {
    return new Promise((resolvePromise, reject) => {
        execFile(GIT, args, { timeout, maxBuffer: 1024 * 1024 }, (error, stdout, stderr) => {
            if (error !== null)
                reject(new Error(`workspace git failed: ${stderr}`.slice(0, 512)));
            else
                resolvePromise(stdout.trim());
        });
    });
}
/**
 * Create an independent writable clone from one immutable source commit.
 * Local object hardlinks and remotes are deliberately removed: the job gets
 * useful history and a branch, but no path that can mutate the source repo.
 * The state root may contain an explicitly approved runner-owned checkpoint
 * used for task continuation. A normal project repository must stay disjoint.
 */
export async function createWorkspaceSnapshot(options) {
    if (!RUN_ID.test(options.runId))
        throw new Error('runId must be lowercase hexadecimal');
    const repository = canonicalDirectory(options.repository, 'repository');
    const stateRoot = canonicalDirectory(options.stateRoot, 'stateRoot');
    if (isWithin(repository, stateRoot) ||
        (isWithin(stateRoot, repository) && options.allowSourceWithinStateRoot !== true)) {
        throw new Error('stateRoot and source repository must be disjoint');
    }
    const revision = await resolveCommit(repository, options.revision);
    const runRoot = join(stateRoot, options.runId);
    const workspace = join(runRoot, 'workspace');
    if (isWithin(repository, runRoot) || isWithin(runRoot, repository)) {
        throw new Error('run workspace and source repository must be disjoint');
    }
    const branch = `agent/${options.runId}`;
    try {
        mkdirSync(runRoot, { mode: 0o700 });
        await git(['clone', '--quiet', '--no-local', '--no-checkout', repository, workspace]);
        await git(['-C', workspace, 'checkout', '--quiet', '-b', branch, revision]);
        await git(['-C', workspace, 'remote', 'remove', 'origin']);
        await git(['-C', workspace, 'config', 'user.name', 'Local B70 Agent']);
        await git(['-C', workspace, 'config', 'user.email', 'local-agent@b70.invalid']);
        await git(['-C', workspace, 'config', 'commit.gpgSign', 'false']);
        await git(['-C', workspace, 'config', 'core.hooksPath', '/dev/null']);
        const checkedOut = await git(['-C', workspace, 'rev-parse', 'HEAD']);
        if (checkedOut !== revision)
            throw new Error('workspace checkout does not match revision');
    }
    catch (error) {
        rmSync(runRoot, { recursive: true, force: true });
        throw error;
    }
    return { path: workspace, revision, runId: options.runId, branch };
}
