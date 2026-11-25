import { useState, type FormEvent } from "react";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { useSignIn } from "@/lib/auth";

/** Owner sign-in: one password, checked by the gateway; on success the ticker switches to live prices. */
export function SignInDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  const [password, setPassword] = useState("");
  const signIn = useSignIn();

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!password) return;
    signIn.mutate(password, {
      onSuccess: () => {
        setPassword("");
        onOpenChange(false);
      },
    });
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => {
        if (!v) {
          setPassword("");
          signIn.reset();
        }
        onOpenChange(v);
      }}
    >
      <DialogContent className="sm:max-w-[420px]">
        <DialogHeader>
          <DialogTitle>Sign in for live prices</DialogTitle>
          <DialogDescription>Owner only. Visitors see end-of-day data.</DialogDescription>
        </DialogHeader>
        <form onSubmit={submit} className="mt-2 grid gap-3">
          <label className="grid gap-1.5 text-[13px] text-muted-foreground">
            Password
            <input
              type="password"
              autoComplete="current-password"
              autoFocus
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              aria-invalid={signIn.isError || undefined}
              aria-describedby={signIn.isError ? "signin-error" : undefined}
              className="h-10 rounded-md border border-line bg-panel px-3 text-[15px] text-foreground outline-none transition-colors focus:border-saffron"
            />
          </label>
          {signIn.isError && (
            <p id="signin-error" role="alert" className="text-[13px] text-down">
              {signIn.error.message === "Wrong password" ? "That password is not right. Try again." : signIn.error.message}
            </p>
          )}
          <button
            type="submit"
            disabled={!password || signIn.isPending}
            className="btn-solid mt-1 h-10 justify-center rounded-md text-[14px] disabled:opacity-50"
          >
            {signIn.isPending ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </DialogContent>
    </Dialog>
  );
}
