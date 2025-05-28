export function sanitizeFilename(name) {
    if (!name || typeof name !== 'string') return 'untitled_movie';
    return name
        .toLowerCase()
        .replace(/\s+/g, '_')
        .replace(/[^a-z0-9_\-\.]/gi, '')
        .replace(/\.+/g, '.')
        .slice(0, 100);
}

export function extractJsonFromString(str) {
    if (typeof str !== 'string') {
        console.warn("extractJsonFromString: input was not a string, returning as is.", str);
        return str;
    }
    const match = str.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    if (match && match[1]) {
        return match[1].trim();
    }
    return str.trim();
}