// distill-web sign-in page: sign in, or create the administrator on the first visit (setup code,
// then name and password). The server keeps the session in an HttpOnly cookie; nothing is stored here.
'use strict';
(function () {
  const byId = (id) => document.getElementById(id);
  let setupCode = '';

  async function call(method, path, payload) {
    const options = {method: method, credentials: 'same-origin', cache: 'no-store'};
    if (payload !== undefined) {
      options.headers = {'Content-Type': 'application/json'};
      options.body = JSON.stringify(payload);
    }
    const response = await fetch(path, options);
    let body = null;
    try { body = await response.json(); } catch (error) { body = {message: 'Invalid answer from the server (HTTP ' + response.status + ').'}; }
    if (!response.ok) throw new Error(body.message || ('HTTP ' + response.status));
    return body;
  }

  function show(view) {
    for (const id of ['form-login', 'form-code', 'form-setup']) byId(id).hidden = id !== view;
    const setup = view !== 'form-login';
    byId('auth-steps').hidden = !setup;
    byId('auth-back').hidden = view !== 'form-setup';
    const dots = byId('auth-steps').children;
    dots[0].className = view === 'form-code' ? 'on' : '';
    dots[1].className = view === 'form-setup' ? 'on' : '';
    if (view === 'form-login') {
      byId('auth-title').textContent = 'Welcome back';
      byId('auth-sub').textContent = 'Sign in as the administrator of this site.';
      byId('auth-foot').textContent = '';
      byId('login-user').focus();
    } else if (view === 'form-code') {
      byId('auth-title').textContent = 'Let’s set up Distill';
      byId('auth-sub').textContent = 'First, the setup code of this server.';
      byId('auth-foot').textContent = 'Step 1 of 2';
      byId('setup-code').focus();
    } else {
      byId('auth-title').textContent = 'Create the administrator';
      byId('auth-sub').textContent = 'The one account that controls this site.';
      byId('auth-foot').textContent = 'Step 2 of 2';
      byId('setup-password').focus();
    }
  }

  function busy(button, on) {
    button.disabled = on;
    button.setAttribute('aria-busy', on ? 'true' : 'false');
  }

  async function submit(event, button, errorBox, work) {
    event.preventDefault();
    errorBox.textContent = '';
    busy(button, true);
    try {
      await work();
    } catch (error) {
      errorBox.textContent = error.message;
    } finally {
      busy(button, false);
    }
  }

  byId('form-login').addEventListener('submit', (event) => submit(event, byId('login-submit'), byId('login-error'), async () => {
    const user = byId('login-user').value.trim();
    const password = byId('login-password').value;
    if (!user || !password) throw new Error('Write the name and the password.');
    try {
      await call('POST', '/api/auth/login', {user: user, password: password});
    } catch (error) {
      byId('login-password').value = '';
      byId('login-password').focus();
      throw error;
    }
    location.replace('/');
  }));

  byId('form-code').addEventListener('submit', (event) => submit(event, byId('code-submit'), byId('code-error'), async () => {
    const code = byId('setup-code').value.trim();
    if (!code) throw new Error('Write the setup code.');
    await call('POST', '/api/auth/setup/check', {code: code});
    setupCode = code;
    byId('setup-code').value = '';
    show('form-setup');
  }));

  byId('form-setup').addEventListener('submit', (event) => submit(event, byId('setup-submit'), byId('setup-error'), async () => {
    const user = byId('setup-user').value.trim();
    const password = byId('setup-password').value;
    if (!user) throw new Error('Choose a name.');
    if (password.length < 12) throw new Error('The password needs at least 12 characters.');
    if (password !== byId('setup-repeat').value) throw new Error('The two passwords are different.');
    await call('POST', '/api/auth/setup', {code: setupCode, user: user, password: password});
    setupCode = '';
    location.replace('/');
  }));

  byId('auth-back').addEventListener('click', () => {
    setupCode = '';
    byId('setup-error').textContent = '';
    show('form-code');
  });

  // Honest note: on plain HTTP across the network the password travels unencrypted.
  function transportNote() {
    const host = location.hostname;
    const local = host === 'localhost' || host === '127.0.0.1' || host === '[::1]' || host === '::1';
    if (location.protocol === 'https:' || local) return;
    const note = byId('auth-transport');
    note.textContent = 'This page uses plain HTTP: fine on your home network, but put the site behind HTTPS before '
      + 'opening it from anywhere else.';
    note.hidden = false;
  }
  transportNote();

  call('GET', '/api/auth/state').then((state) => {
    if (state.signed_in) { location.replace('/'); return; }
    show(state.configured ? 'form-login' : 'form-code');
  }).catch((error) => {
    byId('auth-sub').textContent = 'The server does not answer: ' + error.message;
  });
})();
