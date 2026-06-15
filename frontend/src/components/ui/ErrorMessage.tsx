import { AlertCircle, RefreshCw } from "lucide-react";

interface ErrorMessageProps {
  message: string;
  onRetry?: () => void;
}

export default function ErrorMessage({ message, onRetry }: ErrorMessageProps) {
  return (
    <div className="card p-5 border border-fraud/20">
      <div className="flex items-start gap-3">
        <AlertCircle size={16} className="text-fraud flex-shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-body font-medium text-primary">
            Failed to load data
          </p>
          <p className="text-xs font-mono text-secondary mt-1 break-words">
            {message}
          </p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="btn-ghost text-xs px-3 py-1.5 flex-shrink-0"
          >
            <RefreshCw size={12} />
            Retry
          </button>
        )}
      </div>
    </div>
  );
}